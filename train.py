import torch
import torch.nn as nn
import yaml
import torch.distributions as distributions
import torch.optim as optim
from critic import Criticnet, SmallMLP
from scorenet import Scorenet
import os
import numpy as np 
import matplotlib
from utils import keep_grad, approx_jacobian_trace, exact_jacobian_trace, \
    set_random_seed, get_logger, dict2namespace, get_opt, visualize, langevin_dynamics
import importlib
import argparse
import matplotlib.pyplot as plt
import wandb
from time import time


USE_WANDB = True

device = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')

def setup_wandb(cfg):
    wandb.init(project='prob-learn-proj')

    wandb.config.epochs = cfg.trainer.epochs
    wandb.config.save_dir = cfg.log.save_dir
    wandb.config.lr_scorenet = cfg.trainer.opt_scorenet.lr
    wandb.config.lr_criticnet = cfg.trainer.opt_criticnet.lr
    wandb.config.val_freq = cfg.log.val_freq
    wandb.config.c_iters = cfg.trainer.c_iters
    wandb.config.s_iters = cfg.trainer.s_iters
    wandb.config.batch_size = cfg.data.batch_size
    wandb.config.scorenet_type = cfg.models.scorenet.type
    wandb.config.criticnet_type = cfg.models.criticnet.type
    wandb.config.data_dir = cfg.data.data_dir
    wandb.config.dataset_type = cfg.data.dataset_type
    wandb.config.tr_max_sample_points = cfg.data.tr_max_sample_points
    wandb.config.te_max_sample_points = cfg.data.te_max_sample_points
    wandb.config.sampling_eps = cfg.inference.eps
    wandb.config.sampling_numsteps = cfg.inference.num_steps
    wandb.config.sampling_numpoints = cfg.inference.num_points


    if cfg.models.scorenet.type == 'models.decoders.resnet_add':
        wandb.config.scorenet_nblocks = cfg.models.scorenet.n_blocks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='test_config.yaml',
    help="Config file path")
    
    return parser.parse_args()


def train(args):
    assert os.path.exists(args.cfg)
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    cfg = dict2namespace(cfg)

    set_random_seed(getattr(cfg.trainer, "seed", 666))
    os.makedirs(cfg.log.save_dir, exist_ok=True)

    if USE_WANDB:
        setup_wandb(cfg)
    
    logger = get_logger(logpath=os.path.join(cfg.log.save_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args.cfg)
    
    # sigmas
    if hasattr(cfg.trainer, "sigmas"):
        np_sigmas = cfg.trainer.sigmas
    else:
        sigma_begin = float(cfg.trainer.sigma_begin)
        sigma_end = float(cfg.trainer.sigma_end)
        num_classes = int(cfg.trainer.sigma_num)
        np_sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_classes))

    sigmas = torch.tensor(np.array(np_sigmas)).float().to(device).view(-1, 1)
    
    sigmas = sigmas[-1:]  #TODO: Just with one sigma for now!
    if USE_WANDB:
        wandb.config.sigma = sigmas.item()
    
    if cfg.models.scorenet.type == 'small_mlp':
        score_net = SmallMLP(in_dim=3)
    else:
        score_net = Scorenet()
    print(score_net)

    if cfg.models.criticnet.type == 'small_mlp':
        critic_net = SmallMLP(in_dim=3)
    else:
        critic_net = Criticnet()
    print(critic_net)
    
    critic_net.to(device)
    score_net.to(device)
    
    opt_scorenet, scheduler_scorenet = get_opt(score_net.parameters(), cfg.trainer.opt_scorenet)
    opt_criticnet, scheduler_criticnet = get_opt(critic_net.parameters(), cfg.trainer.opt_scorenet)
    
    itr = 0
    
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']

    for epoch in range(cfg.trainer.epochs):
        for data in train_loader:
            score_net.train()
            critic_net.train()
            opt_scorenet.zero_grad()
            opt_criticnet.zero_grad()

            tr_pts = data['tr_points'].to(device)
            tr_pts.requires_grad_()

            batch_size = tr_pts.size(0)
            
            # Randomly sample sigma
            labels = torch.randint(0, len(sigmas), (batch_size,), device=tr_pts.device)
            used_sigmas = sigmas[labels].float()
            
            perturbed_points = tr_pts + torch.randn_like(tr_pts) * used_sigmas.view(batch_size, 1, 1)

            score_pred = score_net(perturbed_points, used_sigmas)
            critic_output = critic_net(perturbed_points, used_sigmas)

            t1 = (score_pred * critic_output).sum(-1)
            t2 = exact_jacobian_trace(critic_output, perturbed_points)

            stein = t1 + t2
            l2_penalty = (critic_output * critic_output).sum(-1).mean()
            loss = stein.mean()

            cycle_iter = itr % (cfg.trainer.c_iters + cfg.trainer.s_iters)
            
            cpu_loss = loss.detach().cpu().item()
            cpu_t1 = t1.mean().detach().cpu().item()
            cpu_t2 = t2.mean().detach().cpu().item()
            
            if USE_WANDB:
                wandb.log({'epoch': epoch, 'loss_term1': cpu_t1, 'loss_term2': cpu_t2, \
                    'loss': cpu_loss,'itaration': itr})

            if cycle_iter < cfg.trainer.c_iters:
                (-loss + l2_penalty).backward()
                opt_criticnet.step()
                log_message = "Epoch %d itr %d (critic), Loss=%2.5f t1=%2.5f t2=%2.5f" % (epoch, itr, cpu_loss, cpu_t1, cpu_t2)
            else:
                loss.backward()
                opt_scorenet.step()
                log_message = "Epoch %d itr %d (score), Loss=%2.5f t1=%2.5f t2=%2.5f" % (epoch, itr, cpu_loss, cpu_t1, cpu_t2)
            
            if itr % cfg.log.log_freq == 0:
                logger.info(log_message)
            
            if itr % cfg.log.save_freq == 0:
                score_net.cpu()

                torch.save({
                    'args': args,
                    'state_dict': score_net.state_dict(),
                }, os.path.join(cfg.log.save_dir, 'checkpt.pth'))
                
                score_net.to(device)
            
            if itr % cfg.log.viz_freq == 0:
                plt.clf()

                pt_cl, _ = langevin_dynamics(score_net, sigmas, eps=1e-4, num_steps=cfg.inference.num_steps)

                fig, ax = visualize(pt_cl, return_fig=True)

                if USE_WANDB:
                    wandb.log({"langevin_dynamics": wandb.Image(ax)})

                fig_filename = os.path.join(cfg.log.save_dir, 'figs', '{:04d}.png'.format(itr))
                os.makedirs(os.path.dirname(fig_filename), exist_ok=True)
                plt.savefig(fig_filename)
            
            itr += 1


if __name__ == '__main__':
    args = parse_args()
    train(args)