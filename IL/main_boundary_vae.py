import numpy as np
import argparse
import h5py
import os
import pdb
import pickle
import torch
import gym
import math

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from plot_utils import plot_pickle_results

import grid_world as gw
import circle_world as cw
from load_expert_traj import Expert, ExpertHDF5, CircleExpertHDF5
from load_expert_traj import recursively_save_dict_contents_to_group
from itertools import product
from base_models import Policy, Posterior, DiscretePosterior

from utils.logger import Logger, TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network
from boundary_train_vae import VAETrain
def main(args):

    # Create Logger
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor
    vae_train = VAETrain(
        args,
        logger,
        width=11,
        height=15,
        state_size=args.vae_state_size,
        action_size=args.vae_action_size,
        dtype=dtype,
        env_type=args.env_type,
        use_boundary=args.use_boundary
    )

    expert = ExpertHDF5(args.expert_path, args.vae_state_size)
    expert.push(only_coordinates_in_state=True, one_hot_action=True)
    vae_train.set_expert(expert)
    vae_train.get_boundary()
    # expert = Expert(args.expert_path, 2)
    # expert.push()

    if args.run_mode == 'test' or args.run_mode == 'test_goal_pred':
        if args.use_discrete_vae:
            vae_train.vae_model.temperature = 0.1
        assert len(args.checkpoint_path) > 0, \
                'No checkpoint provided for testing'
        vae_train.load_checkpoint(args.checkpoint_path)
        print("Did load models at: {}".format(args.checkpoint_path))
        results_pkl_path = os.path.join(args.results_dir, 'pred_result_cp_1000.pkl')
        vae_train.test_models(expert, results_pkl_path=results_pkl_path,
                            num_test_samples=20)
        plot_pickle_results(results_pkl_path=results_pkl_path, obstacles=vae_train.obstacles, rooms=vae_train.rooms, num_traj_to_plot=20)

    elif args.run_mode == 'train':
        if len(args.finetune_path) > 0:
            vae_train.load_checkpoint(args.finetune_path)
            assert os.path.dirname(os.path.realpath(args.finetune_path)) != \
                    os.path.dirname(os.path.realpath(args.results_dir)), \
                    "Do not save new results in finetune dir."
        vae_train.train(expert, args.num_epochs, args.batch_size)
    else:
        raise ValueError('Incorrect mode to run in.')

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


    parser.add_argument('--cosine_similarity_loss_weight', type=float,
                        default=0, help='Use cosine loss for context.')

    # Arguments for VAE training
    parser.add_argument('--use_discrete_vae', dest='use_discrete_vae',
                        action='store_true', help='Use Discrete VAE.')
    parser.set_defaults(use_discrete_vae=True)

    parser.add_argument('--vae_state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--vae_action_size', type=int, default=4,
                        help='Action size for VAE.')
    parser.add_argument('--vae_context_size', type=int, default=10,
                        help='Context size for VAE.')

    # Temperature
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Discrete VAE temperature')


    # Use features
    parser.add_argument('--use_state_features', dest='use_state_features',
                        action='store_true',
                        help='Use features instead of direct (x,y) values in VAE')
    parser.add_argument('--no-use_state_features', dest='use_state_features',
                        action='store_false',
                        help='Do not use features instead of direct (x,y) ' \
                              'values in VAE')
    parser.set_defaults(use_state_features=False)

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

    # Environment - Grid or Mujoco
    parser.add_argument('--env-type', default='grid_room', 
                        choices=['grid', 'grid_room', 'mujoco', 'gym', 'circle',
                                 'mujoco_custom'],
                        help='Environment type Grid or Mujoco.')

    # Mode to run algorithm
    parser.add_argument('--run_mode', type=str, default='train',
                        choices=['train', 'test',
                                 'train_goal_pred', 'test_goal_pred'],
                        help='Mode to run in.')
    # Expert batch episode length
    parser.add_argument('--episode_len', type=int, default=16,
                        help='Fixed episode length for batch training.')

    parser.add_argument('--lambda_policy', type=float, default=1., help='coefficient for BC loss')
    parser.add_argument('--lambda_kld', type=float, default=10., help='coefficient for KLDistance loss')
    parser.add_argument('--lambda_d_adjacent', type=float, default=1., help='coefficient for adjacent Distance loss')
    parser.add_argument('--warmup_epochs', type=int, default=300)
    
    # model_selection
    parser.add_argument('--use_lstm_transition', action='store_true')
    
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global_args = args
    main(args)
