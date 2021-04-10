import torch
import torch.nn as nn
import yaml
import torch.distributions as distributions
import torch.optim as optim
from critic import Criticnet
from scorenet import Scorenet
import os
from datasets import toy_data
import numpy as np 
import matplotlib
from utils import keep_grad, approx_jacobian_trace, exact_jacobian_trace, \
    set_random_seed, get_logger, dict2namespace, get_opt, visualize_2d, langevin_dynamics_v2, \
    apply_spectral_norm
import importlib
import argparse
import matplotlib.pyplot as plt


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='test_config_toy_2d.yaml',
    help="Config file path")
    
    return parser.parse_args()


def sample_data(data, n_points):
    x = toy_data.inf_train_gen(data, n_points=n_points)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    return x


def train(args):
    assert os.path.exists(args.cfg)
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    cfg = dict2namespace(cfg)
    os.makedirs(cfg.log.save_dir, exist_ok=True)
    
    logger = get_logger(logpath=os.path.join(cfg.log.save_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args.cfg)
    
    
    #sigmas
    if hasattr(cfg.trainer, "sigmas"):
        np_sigmas = cfg.trainer.sigmas
    else:
        sigma_begin = float(cfg.trainer.sigma_begin)
        sigma_end = float(cfg.trainer.sigma_end)
        num_classes = int(cfg.trainer.sigma_num)
        np_sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_classes))

    sigmas = torch.tensor(np.array(np_sigmas)).float().to(device).view(-1, 1)
    
    score_net = Scorenet(in_dim=2)
    critic_net = Criticnet(in_dim=2)
    
    if cfg.models.critic.spectral_norm:
        critic_net.apply(apply_spectral_norm)

    critic_net.to(device)
    score_net.to(device)
    
    opt_scorenet, scheduler_scorenet = get_opt(score_net.parameters(), cfg.trainer.opt_scorenet)
    opt_criticnet, scheduler_criticnet = get_opt(critic_net.parameters(), cfg.trainer.opt_scorenet)
    
    itr = 0

    for epoch in range(cfg.trainer.epochs):
        tr_pts = sample_data('pinwheel', cfg.data.tr_max_sample_points).view(1, -1, 2)
        score_net.train()
        critic_net.train()
        opt_scorenet.zero_grad()
        opt_criticnet.zero_grad()

        #tr_pts = data.to(device)
        #tr_pts = tr_pts.view(1, -1, 2)
        tr_pts.requires_grad_()

        batch_size = tr_pts.size(0)
        
        # Randomly sample sigma
        #labels = torch.randint(0, len(sigmas), (batch_size,), device=tr_pts.device)
        #used_sigmas = sigmas[labels].float()
        
        #perturbed_points = tr_pts + torch.randn_like(tr_pts) * used_sigmas.view(batch_size, 1, 1)
        perturbed_points = tr_pts

        #score_pred = score_net(perturbed_points, used_sigmas)
        score_pred = score_net(perturbed_points)
        
        #critic_output = critic_net(perturbed_points, used_sigmas)
        critic_output = critic_net(perturbed_points)

        t1 = (score_pred * critic_output).sum(-1)
        t2 = exact_jacobian_trace(critic_output, perturbed_points)

        stein = t1 + t2
        l2_penalty = (critic_output * critic_output).sum(-1).mean() * 10.
        loss = stein.mean()

        cycle_iter = itr % (cfg.trainer.c_iters + cfg.trainer.s_iters)
        
        cpu_loss = loss.detach().cpu().item()
        cpu_t1 = t1.mean().detach().cpu().item()
        cpu_t2 = t2.mean().detach().cpu().item()

        if cycle_iter < cfg.trainer.c_iters:
            (-loss + l2_penalty).backward()
            opt_criticnet.step()
            log_message = "Epoch %d itr %d (critic), Loss=%2.5f t1=%2.5f t2=%2.5f" % (epoch, itr, cpu_loss, cpu_t1, cpu_t2)
        else:
            loss.backward()
            opt_scorenet.step()
            log_message = "Epoch %d itr %d (score), Loss=%2.5f t1=%2.5f t2=%2.5f" % (epoch, itr, cpu_loss, cpu_t1, cpu_t2)
        
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

            #pt_cl, _ = langevin_dynamics(score_net, sigmas, dim=2, eps=1e-4, num_steps=cfg.inference.num_steps)
            x_final, _ = langevin_dynamics_v2(score_net, sigmas, dim=2, num_points=cfg.inference.num_points, 
            num_steps=cfg.inference.num_steps, eps=2*1e-5)

            visualize_2d(x_final[0])

            fig_filename = os.path.join(cfg.log.save_dir, 'figs', 'sample-{:04d}.png'.format(itr))
            os.makedirs(os.path.dirname(fig_filename), exist_ok=True)
            plt.savefig(fig_filename)


            # visualize_2d(perturbed_points[0])

            # fig_filename = os.path.join(cfg.log.save_dir, 'figs', 'perturbed-{:04d}.png'.format(itr))
            # os.makedirs(os.path.dirname(fig_filename), exist_ok=True)
            # plt.savefig(fig_filename)
        
        itr += 1


if __name__ == '__main__':
    args = parse_args()
    train(args)