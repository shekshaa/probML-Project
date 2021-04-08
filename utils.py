import torch
import random
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import argparse
import logging 


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_opt(params, cfgopt):
    if cfgopt.type == 'adam':
        optimizer = optim.Adam(params, lr=float(cfgopt.lr),
                               betas=(cfgopt.beta1, cfgopt.beta2),
                               weight_decay=cfgopt.weight_decay)
    elif cfgopt.type == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=float(cfgopt.lr), momentum=cfgopt.momentum)
    else:
        assert 0, "Optimizer type should be either 'adam' or 'sgd'"

    scheduler = None
    scheduler_type = getattr(cfgopt, "scheduler", None)
    if scheduler_type is not None:
        if scheduler_type == 'exponential':
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        elif scheduler_type == 'step':
            step_size = int(getattr(cfgopt, "step_epoch", 500))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)
        elif scheduler_type == 'linear':
            step_size = int(getattr(cfgopt, "step_epoch", 2000))
            final_ratio = float(getattr(cfgopt, "final_ratio", 0.01))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.5))
            duration_ratio = float(getattr(cfgopt, "duration_ratio", 0.45))

            def lambda_rule(ep):
                lr_l = 1.0 - min(1, max(0, ep - start_ratio * step_size) / float(duration_ratio * step_size)) * (1 - final_ratio)
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif scheduler_type == 'cosine_anneal_nocycle':
            final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
            eta_min = float(cfgopt.lr) * final_lr_ratio
            eta_max = float(cfgopt.lr)

            total_epoch = int(getattr(cfgopt, "step_epoch", 2000))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.2))
            T_max = total_epoch * (1 - start_ratio)

            def lambda_rule(ep):
                curr_ep = max(0., ep - start_ratio * total_epoch)
                lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * curr_ep / T_max))
                lr_l = lr / eta_max
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step'"
    return optimizer, scheduler


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx

def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.size(2)):
        fxi = fx[..., i]
        dfxi_dxi = keep_grad(fxi.sum(), x)[..., i]
        vals.append(dfxi_dxi)
    vals = torch.stack(vals, dim=2)
    return vals.sum(dim=2)

def get_prior(batch_size, num_points, inp_dim):
    # -1 to 1, uniform
    return torch.rand(batch_size, num_points, inp_dim) * 2. - 1.

def langevin_dynamics(model, sigmas, num_points=2048, dim=3, eps=2*1e-3, num_steps=10):
    with torch.no_grad():
        x_list = []
        model.eval()
        x = get_prior(1, num_points, dim).cuda()
        x_list.append(x.clone())
        for sigma in sigmas:
            alpha = eps * ((sigma / sigmas[-1]) ** 2)
            for t in range(num_steps):
                z_t = torch.randn_like(x)
                x += torch.sqrt(alpha) * z_t + (alpha / 2.) * model(x, sigma.view(1, -1))
            x_list.append(x.clone())
        return x, x_list

def langevin_dynamics_lsd(f, l=1., e=.01, num_points=2048, n_steps=100, anneal=None):
        x_k = get_prior(1, num_points, 2).cuda()
        # sgld
        if anneal == "lin":
            lrs = list(reversed(np.linspace(e, l, n_steps)))
        elif anneal == "log":
            lrs = np.logspace(np.log10(l), np.log10(e))
        else:
            lrs = [l for _ in range(n_steps)]
        for this_lr in lrs:
            x_k.data += this_lr * f(x_k, torch.tensor(this_lr).view(1, 1).cuda()) + torch.randn_like(x_k) * e
        final_samples = x_k.detach()
        return final_samples



def langevin_dynamics_ebm(model, sigmas, num_points=2048, dim=3, eps=2*1e-3, num_steps=10):
    x_list = []
    model.eval()
    x = get_prior(1, num_points, dim).cuda()
    
    x_list.append(x.clone())
    for sigma in sigmas:
        alpha = eps * ((sigma / sigmas[-1]) ** 2)
        for t in range(num_steps):
            z_t = torch.randn_like(x)
            x_ = x.detach().requires_grad_()
            logp_u = model(x_, sigma.view(1, -1))
            score = keep_grad(logp_u.sum(), x_)
            x += torch.sqrt(alpha) * z_t + (alpha / 2.) * score
        x_list.append(x.clone())
    return x, x_list


def visualize(pts):
    pts = pts.detach().cpu().squeeze().numpy()
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    #plt.show()



def visualize_2d(pts):
    pts = pts.detach().cpu().squeeze().numpy()
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111)
    ax1.scatter(pts[:, 0], pts[:, 1])
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
#     logger.info(filepath)
#     with open(filepath, "r") as f:
#         logger.info(f.read())

#     for f in package_files:
#         logger.info(f)
#         with open(f, "r") as package_f:
#             logger.info(package_f.read())

    return logger
