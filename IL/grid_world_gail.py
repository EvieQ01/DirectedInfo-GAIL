from tqdm import trange
import argparse
import copy
import sys
import os
import pdb
import pickle
import math
import random
import gym
from collections import namedtuple
from itertools import count, product

import numpy as np
import scipy.optimize
from scipy.stats import norm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from models import DiscretePolicy, Policy
from models import DiscretePosterior, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import StateVector, ActionVector
from grid_world import RewardFunction, RewardFunction_SR2, GridWorldReward
from grid_world import ActionBasedGridWorldReward
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5, SeparateRoomTrajExpert
from utils.replay_memory import Memory
from utils.running_state import ZFilter
from utils.torch_utils import clip_grads

from base_gail import BaseGAIL, GAILMLP
from vae import VAE, VAETrain
from utils.logger import Logger, TensorboardXLogger
from utils.rl_utils import epsilon_greedy_linear_decay, epsilon_greedy
from utils.rl_utils import scalar_epsilon_greedy_linear_decay 
from utils.rl_utils import greedy, oned_to_onehot
from utils.rl_utils import get_advantage_for_rewards
from utils.torch_utils import get_weight_norm_for_network
from utils.torch_utils import normal_log_density
from utils.torch_utils import add_scalars_to_summary_writer



def check_args(saved_args, new_args):
    assert saved_args.use_state_features == new_args.use_state_features, \
            'Args do not match - use_state_features'

def load_VAE_model(model_checkpoint_path, new_args):
    '''Load pre-trained VAE model.'''

    checkpoint_dir_path = os.path.dirname(model_checkpoint_path)
    results_dir_path = os.path.dirname(checkpoint_dir_path)

    # Load arguments used to train the model
    saved_args_filepath = os.path.join(results_dir_path, 'args.pkl')
    with open(saved_args_filepath, 'rb') as saved_args_f:
        saved_args = pickle.load(saved_args_f)
        print('Did load saved args {}'.format(saved_args_filepath))

    # check args
    check_args(saved_args, new_args)

    dtype = torch.FloatTensor
    # Use new args to load the previously saved models as well
    if new_args.cuda:
        dtype = torch.cuda.FloatTensor
    logger = TensorboardXLogger(os.path.join(new_args.results_dir, 'log_vae_model'))
    vae_train = VAETrain(
        saved_args,
        logger,
        width=11,
        height=15,
        state_size=saved_args.vae_state_size,
        action_size=saved_args.vae_action_size,
        history_size=saved_args.vae_history_size,
        num_goals=saved_args.vae_goal_size,
        use_rnn_goal_predictor=saved_args.use_rnn_goal,
        dtype=dtype,
        env_type=saved_args.env_type,
        env_name=saved_args.env_name
    )

    vae_train.load_checkpoint(model_checkpoint_path)
    if new_args.cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    vae_train.convert_models_to_type(dtype)
    print("Did load models at: {}".format(model_checkpoint_path))
    return vae_train

def create_result_dirs(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        # Directory for TF logs
        os.makedirs(os.path.join(results_dir, 'log'))
        # Directory for model checkpoints
        os.makedirs(os.path.join(results_dir, 'checkpoint'))

def main(args):
    # Create Logger
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

    # Load finetune args.
    finetune_args = None
    if len(args.finetune_path) > 0:
        finetune_args_path = os.path.dirname(os.path.dirname(args.finetune_path))
        finetune_args_path = os.path.join(finetune_args_path, 'args.pkl')
        assert os.path.exists(finetune_args_path), "Finetune args does not exist."
        with open(finetune_args_path, 'rb') as finetune_args_f:
            finetune_args = pickle.load(finetune_args_f)

    print('Loading expert trajectories ...')
    if 'grid' in args.env_type:
        if args.env_type == 'grid_room':
            expert = SeparateRoomTrajExpert(args.expert_path, args.state_size)
        else:
            expert = ExpertHDF5(args.expert_path, args.state_size)
        expert.push(only_coordinates_in_state=True, one_hot_action=True)
    elif args.env_type == 'mujoco':
        expert = ExpertHDF5(args.expert_path, args.state_size)
        expert.push(only_coordinates_in_state=False, one_hot_action=False)
    print('Expert trajectories loaded.')

    # Load pre-trained VAE model
    vae_train = load_VAE_model(args.vae_checkpoint_path, args)
    vae_train.set_expert(expert)

    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    gail_mlp = GAILMLP(
            args,
            vae_train,
            logger,
            state_size=args.state_size,
            action_size=args.action_size,
            context_size=args.context_size,
            num_goals=args.goal_size,
            history_size=args.history_size,
            dtype=dtype)
    gail_mlp.set_expert(expert)

    if len(args.checkpoint_path) > 0:
        print("Test checkpoint: {}".format(args.checkpoint_path))
        gail_mlp.load_checkpoint_data(args.checkpoint_path)
        results_pkl_path = os.path.join(
                args.results_dir,
                'results_' + os.path.basename(args.checkpoint_path)[:-3] \
                        + 'pkl')
        gail_mlp.get_value_function_for_grid()

        gail_mlp.train_gail(
                1,
                results_pkl_path,
                gen_batch_size=512,
                train=False)
        print("Did save results to: {}".format(results_pkl_path))
        return

    if len(args.finetune_path) > 0:
        # -4 removes .pth from finetune path
        checkpoint_name = os.path.basename(args.finetune_path)[:-4]
        # Create results directory for finetune results.
        results_dir = os.path.join(args.results_dir,
                                   'finetune_' + checkpoint_name)
        create_result_dirs(results_dir)
        # Set new Tensorboard logger for finetune results.
        logger = TensorboardXLogger(os.path.join(results_dir, 'log'))
        gail_mlp.logger = logger

        print("Finetune checkpoint: {}".format(args.finetune_path))
        gail_mlp.load_checkpoint_data(args.finetune_path)
        gail_mlp.get_value_function_for_grid()
        gail_mlp.train_gail(
                args.num_epochs,
                os.path.join(results_dir, 'results.pkl'),
                gen_batch_size=args.batch_size,
                train=True)
        return

    if args.init_from_vae:
        print("Will load generator and posterior from pretrianed VAE.")
        gail_mlp.load_weights_from_vae()

    results_path = os.path.join(args.results_dir, 'results.pkl')
    gail_mlp.train_gail(
            args.num_epochs,
            os.path.join(args.results_dir, 'results.pkl'),
            gen_batch_size=args.batch_size,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal GAIL using MLP.')
    parser.add_argument('--expert_path', default="L_expert_trajectories/",
                        help='path to the expert trajectory files')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    # Environment parameters
    parser.add_argument('--state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--action_size', type=int, default=4,
                        help='Action size for VAE.')
    parser.add_argument('--history_size', type=int, default=1,
                        help='State history size to use in VAE.')
    parser.add_argument('--context_size', type=int, default=1,
                        help='Context size for VAE.')
    parser.add_argument('--goal_size', type=int, default=1,
                        help='Goal size for VAE.')

    # RL parameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae (default: 0.95)')

    parser.add_argument('--lambda_posterior', type=float, default=1.0,
                        help='Parameter to scale MI loss from the posterior.')
    parser.add_argument('--lambda_goal_pred_reward', type=float, default=1.0,
                        help='Reward scale for goal prediction reward from RNN.')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='gae (default: 3e-4)')
    parser.add_argument('--posterior_learning_rate', type=float, default=3e-4,
                        help='VAE posterior lr (default: 3e-4)')
    parser.add_argument('--gen_learning_rate', type=float, default=3e-4,
                        help='Generator lr (default: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size (default: 2048)')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of episodes (default: 500)')
    parser.add_argument('--max_ep_length', type=int, default=1000,
                        help='maximum episode length.')

    parser.add_argument('--optim_epochs', type=int, default=5,
                        help='number of epochs over a batch (default: 5)')
    parser.add_argument('--optim_batch_size', type=int, default=64,
                        help='batch size for epochs (default: 64)')
    parser.add_argument('--num_expert_trajs', type=int, default=5,
                        help='number of expert trajectories in a batch.')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    # Log interval
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Interval between training status logs')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Interval between saving policy weights')
    parser.add_argument('--entropy_coeff', type=float, default=0.0,
                        help='coefficient for entropy cost')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='Clipping for PPO grad')

    # Path to pre-trained VAE model
    parser.add_argument('--vae_checkpoint_path', type=str, required=True,
                        help='Path to pre-trained VAE model.')
    # Path to store training results in
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to store results in.')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint path to load pre-trained models.')
    parser.add_argument('--finetune_path', type=str, default='',
                        help='pre-trained models to finetune.')

    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Disable CUDA training')
    parser.set_defaults(cuda=False)

    # Use features
    parser.add_argument('--use_state_features', dest='use_state_features',
                        action='store_true',
                        help='Use features instead of direct (x,y) values in VAE')
    parser.add_argument('--no-use_state_features', dest='use_state_features',
                        action='store_false',
                        help='Do not use features instead of direct (x,y) ' \
                            'values in VAE')
    parser.set_defaults(use_state_features=False)

    # Use reparameterization for posterior training.
    parser.add_argument('--use_reparameterize', dest='use_reparameterize',
                        action='store_true',
                        help='Use reparameterization during posterior training ' \
                            'values in VAE')
    parser.add_argument('--no-use_reparameterize', dest='use_reparameterize',
                        action='store_false',
                        help='Use reparameterization during posterior training ' \
                            'values in VAE')
    parser.set_defaults(use_reparameterize=False)

    parser.add_argument('--flag_true_reward', type=str, default='grid_reward',
                        choices=['grid_reward', 'action_reward'],
                        help='True reward type to use.')
    parser.add_argument('--disc_reward', choices=['no_log', 'log_d', 'log_1-d'],
                        default='log_d',
                        help='Discriminator reward to use.')

    parser.add_argument('--use_value_net', dest='use_value_net',
                        action='store_true',
                        help='Use value network.')
    parser.add_argument('--no-use_value_net', dest='use_value_net',
                        action='store_false',
                        help='Don\'t use value network.')
    parser.set_defaults(use_value_net=True)

    parser.add_argument('--init_from_vae', dest='init_from_vae',
                        action='store_true',
                        help='Init policy and posterior from vae.')
    parser.add_argument('--no-init_from_vae', dest='init_from_vae',
                        action='store_false',
                        help='Don\'t init policy and posterior from vae.')
    parser.set_defaults(init_from_vae=True)

    # Environment - Grid or Mujoco
    parser.add_argument('--env-type', default='grid',
                        choices=['grid', 'grid_room', 'mujoco'],
                        help='Environment type Grid or Mujoco.')
    parser.add_argument('--env-name', default=None,
                        help='Environment name if Mujoco.')

    # Action - discrete or continuous
    parser.add_argument('--discrete_action', dest='discrete_action',
                        action='store_true',
                        help='actions are discrete, use BCE loss')
    parser.add_argument('--continuous_action', dest='discrete_action',
                        action='store_false',
                        help='actions are continuous, use MSE loss')
    parser.set_defaults(discrete_action=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    create_result_dirs(args.results_dir)

    # Save runtime arguments to pickle file
    args_pkl_filepath = os.path.join(args.results_dir, 'args.pkl')
    with open(args_pkl_filepath, 'wb') as args_pkl_f:
        pickle.dump(args, args_pkl_f, protocol=2)

    main(args)
