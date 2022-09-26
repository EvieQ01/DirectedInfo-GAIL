
import numpy as np
import argparse
import h5py
import os
import pdb
import torch
import math

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from boundary_utils import get_boundary_from_all_traj
import grid_world as gw
from load_expert_traj import Expert, ExpertHDF5, CircleExpertHDF5

from utils.logger import Logger, TensorboardXLogger
import wandb
import time
def main(args):

    # Create Logger
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    expert = ExpertHDF5(args.expert_path, args.vae_state_size)
    expert.push(only_coordinates_in_state=True, one_hot_action=True)
    traj_expert = expert.sample_all()
    state_expert, action_expert, c_expert, _ = traj_expert
    boundary_list = get_boundary_from_all_traj(state_expert)     

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VAE Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--checkpoint_every_epoch', type=int, default=10,
                        help='Save models after ever N epochs.')
    # Run on GPU
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Disable CUDA training')
    parser.set_defaults(cuda=False)

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging ' \
                              'training status')
    parser.add_argument('--expert-path', default='./h5_trajs/room_trajs/traj_len_16',
                        metavar='G',
                        help='path to the expert trajectory files')

    parser.add_argument('--vae_state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--vae_action_size', type=int, default=4,
                        help='Action size for VAE.')
    parser.add_argument('--vae_context_size', type=int, default=10,
                        help='Context size for VAE.')

    # Temperature
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Discrete VAE temperature')


    # Use boundary
    parser.add_argument('--use_boundary', action='store_true')
    parser.set_defaults(use_boundary=True)

    # Logging flags
    parser.add_argument('--log_gradients_tensorboard',
                        dest='log_gradients_tensorboard', action='store_true',
                        help='Log network weights and grads in tensorboard.')
    parser.add_argument('--no-log_gradients_tensorboard',
                        dest='log_gradients_tensorboard', action='store_true',
                        help='Log network weights and grads in tensorboard.')
    parser.set_defaults(log_gradients_tensorboard=True)

    # Results dir
    parser.add_argument('--results_dir', type=str, default='./results/vae/room_traj/discrete/traj_room_centre_len_16',
                        help='Directory to save final results in.')
    # Checkpoint directory to load pre-trained models.
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint path to load pre-trained models.')
    parser.add_argument('--finetune_path', type=str, default='',
                        help='pre-trained models to finetune.')
    parser.add_argument('--results_pkl_path', default='./results/test_predict/room_traj/context_4/pred_result_cp_1000.pth')

    # Action - discrete or continuous
    parser.add_argument('--discrete_action', dest='discrete_action',
                        action='store_true',
                        help='actions are discrete, use BCE loss')
    parser.add_argument('--continuous_action', dest='discrete_action',
                        action='store_false',
                        help='actions are continuous, use MSE loss')
    parser.set_defaults(discrete_action=True)
    # training hyperparams
    
    # debug
    parser.add_argument('--debug', action='store_true', help='whether print out debugging message')
    parser.add_argument('--wandb', action='store_true', help='whether save on wandb')
    args = parser.parse_args()
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    if args.wandb:
        wandb.init(project="boundary_grid_room", entity="evieq01")
        wandb.run.name = f'{now}'
        wandb.config.update(args)

    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global_args = args
    main(args)
