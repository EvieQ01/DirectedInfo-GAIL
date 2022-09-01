import numpy as np
import pdb
import gym
from tqdm import trange
import argparse
import copy
import h5py
import math
import os
import pickle
import random
from itertools import count, product

from scipy.stats import norm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from models import Policy, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import RewardFunction, RewardFunction_SR2, GridWorldReward
from grid_world import ActionBasedGridWorldReward
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5
from utils.replay_memory import Memory
from utils.running_state import ZFilter
from utils.torch_utils import clip_grads

from vae import VAE, VAETrain
from utils.logger import Logger, TensorboardXLogger
from models import DiscretePolicy, Policy
from models import DiscretePosterior, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import StateVector, ActionVector
from grid_world import RewardFunction, RewardFunction_SR2, GridWorldReward
from grid_world import ActionBasedGridWorldReward
from grid_world import create_obstacles, obstacle_movement, sample_start
from utils.rl_utils import epsilon_greedy_linear_decay, epsilon_greedy
from utils.rl_utils import scalar_epsilon_greedy_linear_decay 
from utils.rl_utils import greedy, oned_to_onehot
from utils.rl_utils import get_advantage_for_rewards
from utils.torch_utils import get_weight_norm_for_network
from utils.torch_utils import normal_log_density
from utils.torch_utils import add_scalars_to_summary_writer

class BaseGAIL(object):
    def __init__(self,
                 args,
                 logger,
                 state_size=2,
                 action_size=4,
                 context_size=1,
                 num_goals=4,
                 history_size=1,
                 dtype=torch.FloatTensor):
        self.args = args
        self.logger = logger

        self.state_size = state_size
        self.action_size = action_size
        self.history_size = history_size
        self.context_size = context_size
        self.num_goals = num_goals
        self.dtype = dtype
        self.train_step_count, self.gail_step_count = 0, 0
        self.env_type = args.env_type

        self.policy_net = None
        self.old_policy_net = None
        self.value_net = None
        self.reward_net = None

        self.opt_policy, self.opt_reward, self.opt_value = None, None, None

        self.transition_func, self.true_reward = None, None
        self.expert = None
        self.obstacles, self.set_diff = None, None

    def create_environment(self):
        self.width, self.height = 21, 21
        self.transition_func = TransitionFunction(self.width,
                                                  self.height,
                                                  obstacle_movement)
    def set_expert(self, expert):
        assert self.expert is None, "Trying to set non-None expert"
        self.expert = expert
        assert expert.obstacles is not None, "Obstacles cannot be None"
        self.obstacles = expert.obstacles
        assert expert.set_diff is not None, "set_diff in h5 file cannot be None"
        self.set_diff = expert.set_diff

        # TODO: Hardcoded for now remove this, load it from the expert trajectory
        # file. Also, since the final state is not in expert trajectory we append
        # very next states as goal as well. Else reward is sparse.

        if self.args.flag_true_reward == 'grid_reward':
            self.true_reward = GridWorldReward(self.width,
                                               self.height,
                                               None,  # No default goals
                                               self.obstacles)
        elif self.args.flag_true_reward == 'action_reward':
            self.true_reward = ActionBasedGridWorldReward(
                    self.width, self.height, None, self.obstacles)


    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0).type(self.dtype)
        action, _, _ = self.policy_net(Variable(state))
        return action

    def get_state_features(self, state_obj, use_state_features):
        '''Get state features.

        state_obj: State object.
        '''
        if use_state_features:
            feat = np.array(state_obj.get_features(), dtype=np.float32)
        else:
            feat = np.array(state_obj.coordinates, dtype=np.float32)
        return feat

    def sample_start_state(self):
        '''Randomly sample start state.'''
        start_loc = sample_start(self.set_diff)
        return State(start_loc, self.obstacles)

    def checkpoint_data_to_save(self):
        raise ValueError("Subclass should override.")

    def load_checkpoint_data(self, checkpoint_path):
        raise ValueError("Subclass should override.")

    def model_checkpoint_filepath(self, epoch):
        checkpoint_dir = os.path.join(self.args.results_dir, 'checkpoint')
        return os.path.join(checkpoint_dir, 'cp_{}.pth'.format(epoch))

    def expand_states_numpy(self, states, history_size):
        if self.history_size == 1:
            return states

        N, C = states.shape
        expanded_states = -1*np.ones((N, C*history_size), dtype=np.float32)
        for i in range(N):
            # Append states to the right
            expanded_states[i, -C:] = states[i,:]
            # Copy C:end state values from i-1 to 0:End-C in i
            if i > 0:
                expanded_states[i, :-C] = expanded_states[i-1, C:]

        return expanded_states

class GAILMLP(BaseGAIL):
    def __init__(self,
            args,
            vae_train,
            logger,
            state_size=2,
            action_size=4,
            context_size=1,
            num_goals=4,
            history_size=1,
            dtype=torch.FloatTensor):
        super(GAILMLP, self).__init__(args,
                logger,
                state_size=state_size,
                action_size=action_size,
                context_size=context_size,
                num_goals=num_goals,
                history_size=history_size,
                dtype=dtype)

        self.vae_train = vae_train
        policy1_state_size = state_size * history_size #\
           # if vae_train.vae_model.use_history_in_policy else state_size

        policy_klass = DiscretePolicy if args.discrete_action else Policy
        self.policy_net = policy_klass(
                state_size=policy1_state_size + args.context_size, # hard coding
                action_size=vae_train.vae_model.policy.action_size,
                latent_size=0,
                output_size=vae_train.vae_model.policy.output_size,
                output_activation=None)

        self.old_policy_net = policy_klass(
                state_size=policy1_state_size + args.context_size, # hard coding
                action_size=vae_train.vae_model.policy.action_size,
                latent_size=0,
                output_size=vae_train.vae_model.policy.output_size,
                output_activation=None)
        
        if args.use_value_net:
            # context_size contains num_goals
            self.value_net = Value(state_size * history_size + context_size,
                                   hidden_size=64)

        # Reward net is the discriminator network.
        self.reward_net = Reward(state_size * history_size ,
                                 action_size,
                                 context_size,
                                 hidden_size=64)

        if vae_train.args.use_discrete_vae:
            self.posterior_net = DiscretePosterior(
                    state_size=state_size*history_size,
                    action_size=vae_train.vae_model.posterior.action_size,
                    latent_size=vae_train.vae_model.posterior.latent_size,
                    hidden_size=vae_train.vae_model.posterior.hidden_size,
                    output_size=vae_train.args.vae_context_size)
        else:
            self.posterior_net = Posterior(state_size * history_size,
                                           0,
                                           context_size,
                                           hidden_size=64)

        self.opt_policy = optim.Adam(self.policy_net.parameters(),
                                     lr=args.gen_learning_rate)
        self.opt_reward = optim.Adam(self.reward_net.parameters(),
                                     lr=args.learning_rate)
        self.opt_posterior = optim.Adam(self.posterior_net.parameters(),
                                        lr=args.posterior_learning_rate)
        if args.use_value_net:
            self.opt_value = optim.Adam(self.value_net.parameters(),
                                        lr=args.learning_rate)

        self.transition_func, self.true_reward = None, None
        self.create_environment(args.env_type, args.env_name)
        self.expert = None
        self.obstacles, self.set_diff = None, None

    def convert_models_to_type(self, dtype):
        self.vae_train.convert_models_to_type(dtype)
        self.policy_net = self.policy_net.type(dtype)
        self.old_policy_net = self.old_policy_net.type(dtype)
        if self.value_net is not None:
            self.value_net = self.value_net.type(dtype)
        self.reward_net = self.reward_net.type(dtype)
        self.posterior_net = self.posterior_net.type(dtype)

    def create_environment(self, env_type, env_name=None):
        if 'grid' in env_type:
            self.transition_func = TransitionFunction(self.vae_train.width,
                                                      self.vae_train.height,
                                                      obstacle_movement)
        elif env_type == 'mujoco':
            assert(env_name is not None)
            self.env = gym.make(env_name)

    def select_action(self, x_var, goal_var):
        """Select action using policy net."""
        assert self.args.discrete_action, "Only implements discrete action."
        inp = torch.cat((x_var, goal_var), dim=1)
        # return self.policy_net.select_action(inp)
        return self.policy_net(inp)

    def get_state_features(self, state_obj, use_state_features):
        if use_state_features:
            feat = np.array(state_obj.get_features(), dtype=np.float32)
        else:
            feat = np.array(state_obj.coordinates, dtype=np.float32)
        return feat

    def get_history_features(self, state, use_velocity=False):
        '''
        state: Numpy array of shape (N, H, F)
        '''
        if use_velocity:
            _, history_size, state_size = state.shape
            new_state = np.zeros(state.shape)

            state_int = np.array(state, dtype=np.int32)
            for t in range(history_size-1):
                minus_one_idx = state_int[:, t, 0] == -1
                new_state[minus_one_idx, t, :] = 0.0
                one_idx = (state_int[:, t, 0] != -1)
                new_state[one_idx, t, :] = \
                        state[one_idx, t+1, :] - state[one_idx, t, :]

            new_state[:, history_size-1, :] = state[:, history_size-1, :]

            return new_state
        else:
            return state

    def get_c_for_traj(self, state_arr, action_arr, c_arr):
        '''Get c[1:T] for given trajectory.'''
        batch_size, episode_len = state_arr.shape[0], state_arr.shape[1]
        history_size = self.history_size

        # Use the Q-network (RNN) to predict goal.
        pred_goal = None
        if self.vae_train.use_rnn_goal_predictor:
            pred_goal, _ = self.vae_train.predict_goal(
                state_arr, action_arr, c_arr, None, self.num_goals)

        if self.args.env_type == 'grid_room':
            true_goal_numpy = np.copy(c_arr)
        else:
            true_goal_numpy = np.zeros((c_arr.shape[0], self.num_goals))
            true_goal_numpy[np.arange(c_arr.shape[0]), c_arr[:, 0]] = 1
        true_goal = Variable(torch.from_numpy(true_goal_numpy).type(self.dtype))

        action_var = Variable(
                torch.from_numpy(action_arr).type(self.dtype))

        # Context output from the VAE encoder
        pred_c_arr = -1 * np.ones((
            batch_size,
            episode_len + 1,
            self.vae_train.vae_model.posterior_latent_size))

        if 'grid' in self.args.env_type:
            x_state_obj = StateVector(state_arr[:, 0, :], self.obstacles)
            x_feat = self.vae_train.get_state_features(
                    x_state_obj,
                    self.args.use_state_features)
        elif self.args.env_type == 'mujoco':
            x_feat = state_arr[:, 0, :]
            dummy_state = self.env.reset()
            self.env.env.set_state(np.concatenate(
                (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])
            dummy_state = x_feat
        else:
            raise ValueError('Incorrect env type: {}'.format(
                self.args.env_type))

        # x is (N, F)
        x = x_feat

        # Add history to state
        if self.history_size > 1:
            x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                  dtype=np.float32)
            x_hist[:, history_size - 1, :] = x_feat
            x = self.vae_train.get_history_features(x_hist)

        for t in range(episode_len):
            c = pred_c_arr[:, t, :]
            x_var = Variable(torch.from_numpy(
                x.reshape((batch_size, -1))).type(self.dtype))

            # Append 'c' at the end.
            if self.vae_train.use_rnn_goal_predictor:
                c_var = torch.cat([
                    final_goal,
                    Variable(torch.from_numpy(c).type(self.dtype))], dim=1)
            else:
                c_var = Variable(torch.from_numpy(c).type(self.dtype))
                if len(true_goal.size()) == 2:
                    c_var = torch.cat([true_goal, c_var], dim=1)
                elif len(true_goal.size()) == 3:
                    c_var = torch.cat([true_goal[:, t, :], c_var], dim=1)
                else:
                    raise ValueError("incorrect true goal size")

            c_next = self.vae_train.get_context_at_state(x_var, c_var)
            pred_c_arr[:, t+1, :] = c_next.data.cpu().numpy()

            if history_size > 1:
                x_hist[:, :(history_size-1), :] = x_hist[:, 1:, :]

            if 'grid' in self.args.env_type:
                if t < episode_len-1:
                    next_state = StateVector(state_arr[:, t+1, :],
                                             self.obstacles)
                else:
                    break

                if history_size > 1:
                    x_hist[:, history_size-1] = self.get_state_features(
                            next_state, self.args.use_state_features)
                    x = self.vae_train.get_history_features(x_hist)
                else:
                    x[:] = self.get_state_features(next_state,
                                                   self.args.use_state_features)
            else:
                raise ValueError("Not implemented yet.")

        return pred_c_arr, pred_goal

    def checkpoint_data_to_save(self):
        value_net_state_dict = None if self.value_net is None else \
                self.value_net.state_dict()
        return {
                'policy': self.policy_net.state_dict(),
                'posterior': self.posterior_net.state_dict(),
                'reward': self.reward_net.state_dict(),
                'value': value_net_state_dict,
        }

    def load_checkpoint_data(self, checkpoint_path):
        assert os.path.exists(checkpoint_path), \
                'Checkpoint path does not exists {}'.format(checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(checkpoint_data['policy'])
        self.posterior_net.load_state_dict(checkpoint_data['posterior'])
        self.reward_net.load_state_dict(checkpoint_data['reward'])
        if checkpoint_data.get('value') is not None:
            self.value_net.load_state_dict(checkpoint_data['value'])

    def load_weights_from_vae(self):
        # deepcopy from vae
        # self.policy_net = copy.deepcopy(self.vae_train.vae_model.policy)
        # self.old_policy_net = copy.deepcopy(self.vae_train.vae_model.policy)
        # self.posterior_net = copy.deepcopy(self.vae_train.vae_model.posterior)

        # re-initialize optimizers
        # self.opt_policy = optim.Adam(self.policy_net.parameters(),
        #                             lr=self.args.learning_rate)
        # self.opt_posterior = optim.Adam(self.posterior_net.parameters(),
        #                                lr=self.args.posterior_learning_rate)
        pass


    def set_expert(self, expert):
        assert self.expert is None, "Trying to set non-None expert"
        self.expert = expert
        assert expert.obstacles is not None, "Obstacles cannot be None"
        self.obstacles = expert.obstacles
        assert expert.set_diff is not None, "set_diff in h5 file cannot be None"
        self.set_diff = expert.set_diff

    def get_discriminator_reward(self, x, a, cvar):
        '''Get discriminator reward.'''
        disc_reward = float(self.reward_net(torch.cat(
            (x,
             Variable(torch.from_numpy(oned_to_onehot(
                 a, self.action_size)).unsqueeze(0)).type(self.dtype),
             cvar), 1)).data.cpu().numpy()[0,0])

        if self.args.disc_reward == 'log_d':
            if disc_reward < 1e-5:
                disc_reward += 1e-5
            disc_reward = -math.log(disc_reward)
        elif self.args.disc_reward == 'log_1-d':
            if disc_reward >= 1.0:
                disc_reward = 1.0 - 1e-5
            disc_reward = math.log(1 - disc_reward)
        elif self.args.disc_reward == 'no_log':
            disc_reward = -disc_reward
        else:
            raise ValueError("Incorrect Disc reward type: {}".format(
                self.args.disc_reward))
        return disc_reward

    def get_posterior_reward(self, x, c, next_ct, goal_var=None):
        '''Get posterior reward.'''
        if goal_var is not None:
            c = torch.cat([goal_var, c], dim=1)

        if self.vae_train.args.use_discrete_vae:
            logits = self.posterior_net(torch.cat((x, c), 1))
            _, label = torch.max(next_ct, 1)
            posterior_reward_t = F.cross_entropy(logits, label)
        else:
            mu, sigma = self.posterior_net(torch.cat((x, c), 1))
            mu = mu.data.cpu().numpy()[0,0]
            sigma = np.exp(0.5 * sigma.data.cpu().numpy()[0,0])

            # TODO: should ideally be logpdf, but pdf may work better. Try both.
            # use norm.logpdf if flag else use norm.pdf
            use_log_rewards = True
            reward_func = norm.logpdf if use_log_rewards else norm.pdf
            scale = sigma if self.args.use_reparameterize else 0.1
            # use fixed std if not using reparameterize otherwise use sigma.
            posterior_reward_t = reward_func(next_ct, loc=mu, scale=0.1)[0]

        return posterior_reward_t.data.cpu().numpy()[0]

    def get_value_function_for_grid(self):
        from grid_world import create_obstacles, obstacle_movement, sample_start
        '''Get value function for different locations in grid.'''
        grid_width, grid_height = self.vae_train.width, self.vae_train.height
        obstacles, rooms, room_centres = create_obstacles(
                grid_width,
                grid_height,
                env_name='room',
                room_size=3)
        valid_positions = list(set(product(tuple(range(0, grid_width)),
                                    tuple(range(0, grid_height))))
                                    - set(obstacles))
        values_for_goal = -1*np.ones((4, grid_height, grid_width))
        for pos in valid_positions:
            # hardcode number of goals
            for goal_idx in range(4):
                goal_arr = np.zeros((4))
                goal_arr[goal_idx] = 1
                value_tensor = torch.Tensor(np.hstack(
                    [np.array(pos), goal_arr])[np.newaxis, :])
                value_var = self.value_net(Variable(value_tensor))
                values_for_goal[goal_idx, grid_height-pos[1], pos[0]] = \
                        value_var.data.cpu().numpy()[0, 0]
        for g in range(4):
            value_g = values_for_goal[g]
            print("Value for goal: {}".format(g))
            print(np.array_str(
                value_g, precision=2, suppress_small=True, max_line_width=200))

    def get_discriminator_reward_for_grid(self):
        from grid_world import create_obstacles, obstacle_movement, sample_start
        '''Get value function for different locations in grid.'''
        grid_width, grid_height = self.vae_train.width, self.vae_train.height
        obstacles, rooms, room_centres = create_obstacles(
                grid_width,
                grid_height,
                env_name='room',
                room_size=3)
        valid_positions = list(set(product(tuple(range(0, grid_width)),
                                    tuple(range(0, grid_height))))
                                    - set(obstacles))

        reward_for_goal_action = -1*np.ones((4, 4, grid_height, grid_width))
        for pos in valid_positions:
            for action_idx in range(4):
                action_arr = np.zeros((4))
                action_arr[action_idx] = 1
                for goal_idx in range(4):
                    goal_arr = np.zeros((4))
                    goal_arr[goal_idx] = 1
                    inp_tensor = torch.Tensor(np.hstack(
                        [np.array(pos), action_arr, goal_arr])[np.newaxis, :])
                    reward = self.reward_net(Variable(inp_tensor))
                    reward = float(reward.data.cpu().numpy()[0, 0])
                    if self.args.disc_reward == 'log_d':
                        reward = -math.log(reward)
                    elif self.args.disc_reward == 'log_1-d':
                        reward = math.log(1.0 - reward)
                    elif self.args.disc_reward == 'no_log':
                        reward = reward
                    else:
                        raise ValueError("Incorrect disc_reward type")
                    
                    reward_for_goal_action[action_idx,
                                           goal_idx,
                                           grid_height-pos[1],
                                           pos[0]] = reward

        for g in range(4):
            for a in range(4):
                reward_ag = reward_for_goal_action[a, g]
                print("Reward for action: {} goal: {}".format(a, g))
                print(np.array_str(reward_ag,
                                   precision=2,
                                   suppress_small=True,
                                   max_line_width=200))


    def update_posterior_net(self, state_var, c_var, next_c_var, goal_var=None):
        if goal_var is not None:
            c_var = torch.cat([goal_var, c_var], dim=1)

        if self.vae_train.args.use_discrete_vae:
            # pdb.set_trace()
            logits = self.posterior_net(torch.cat((state_var, c_var), 1))
            _, label = torch.max(next_c_var, 1)
            posterior_loss = F.cross_entropy(logits, label)
        else:
            mu, logvar = self.posterior_net(torch.cat((state_var, c_var), 1))
            posterior_loss = F.mse_loss(mu, next_c_var)
        return posterior_loss

    def update_params_for_batch(self,
                                states,
                                actions,
                                latent_c,
                                latent_next_c,
                                targets,
                                advantages,
                                expert_states,
                                expert_actions,
                                expert_latent_c,
                                optim_batch_size,
                                optim_batch_size_exp,
                                optim_iters,
                                goal=None,
                                expert_goal=None):
        '''Update parameters for one batch of data.

        Update the policy network, discriminator (reward) network and the
        posterior network here.
        '''
        args, dtype = self.args, self.dtype
        curr_id, curr_id_exp = 0, 0
        for optim_idx in range(optim_iters):
            curr_batch_size = min(optim_batch_size, actions.size(0) - curr_id)
            curr_batch_size_exp = min(optim_batch_size_exp,
                                      expert_actions.size(0) - curr_id_exp)
            start_idx, end_idx = curr_id, curr_id + curr_batch_size

            state_var = Variable(states[start_idx:end_idx])
            action_var = Variable(actions[start_idx:end_idx])
            latent_c_var = Variable(latent_c[start_idx:end_idx])
            latent_next_c_var = Variable(latent_next_c[start_idx:end_idx])
            advantages_var = Variable(advantages[start_idx:end_idx])
            goal_var = None
            if goal is not None:
                goal_var = Variable(goal[start_idx:end_idx])

            start_idx, end_idx = curr_id_exp, curr_id_exp + curr_batch_size_exp
            expert_state_var = Variable(expert_states[start_idx:end_idx])
            expert_action_var = Variable(expert_actions[start_idx:end_idx])
            expert_latent_c_var = Variable(expert_latent_c[start_idx:end_idx])
            expert_goal_var = None
            if expert_goal is not None:
                expert_goal_var = Variable(expert_goal[start_idx:end_idx])

            if optim_idx % 1 == 0:
                # ==== Update reward net ====
                self.opt_reward.zero_grad()

                # Backprop with expert demonstrations
                                #    expert_goal_var,
                expert_output = self.reward_net(
                        torch.cat((expert_state_var,
                                   expert_action_var,
                                   expert_latent_c_var), 1))
                expert_disc_loss = F.binary_cross_entropy(
                        expert_output,
                        Variable(torch.zeros(expert_action_var.size(0), 1)).type(
                            dtype))
                expert_disc_loss.backward()

                # Backprop with generated demonstrations
                # latent_next_c_var is actual c_t, latent_c_var is c_{t-1}
                                #    goal_var,
                gen_output = self.reward_net(
                        torch.cat((state_var,
                                   action_var,
                                   latent_next_c_var), 1))
                gen_disc_loss = F.binary_cross_entropy(
                        gen_output,
                        Variable(torch.ones(action_var.size(0), 1)).type(dtype))
                gen_disc_loss.backward()

                self.opt_reward.step()
                # ==== END ====

            # Add loss scalars.
            add_scalars_to_summary_writer(
                self.logger.summary_writer,
                'loss/discriminator',
                {
                  'total': expert_disc_loss.data.item() + gen_disc_loss.data.item(),
                  'expert': expert_disc_loss.data.item(),
                  'gen': gen_disc_loss.data.item(),
                  },
                self.gail_step_count)
            reward_l2_norm, reward_grad_l2_norm = \
                              get_weight_norm_for_network(self.reward_net)
            self.logger.summary_writer.add_scalar('weight/discriminator/param',
                                                  reward_l2_norm,
                                                  self.gail_step_count)
            self.logger.summary_writer.add_scalar('weight/discriminator/grad',
                                                  reward_grad_l2_norm,
                                                  self.gail_step_count)

            # ==== Update posterior net ====
            self.opt_posterior.zero_grad()
            posterior_loss = self.update_posterior_net(state_var,
                                                       latent_c_var,
                                                       latent_next_c_var,
                                                       goal_var=goal_var) # TODO goal_var
            posterior_loss.backward()
            self.opt_posterior.step()
            self.logger.summary_writer.add_scalar('loss/posterior',
                                                  posterior_loss.data.item(),
                                                  self.gail_step_count)
            # ==== END ====

            # compute old and new action probabilities
            # if self.args.use_goal_in_policy: # 
            #     action_means, action_log_stds, action_stds = self.policy_net(
            #             torch.cat((state_var, goal_var), 1))
            #     action_means_old, action_log_stds_old, action_stds_old = \
            #             self.old_policy_net(
            #                     torch.cat((state_var, goal_var), 1))
            # else: # change policy net input
            # action_means, action_log_stds, action_stds = self.policy_net(
            #         torch.cat((state_var, latent_next_c_var), 1))
            # action_means_old, action_log_stds_old, action_stds_old = \
            #         self.old_policy_net(
            #                 torch.cat((state_var, latent_next_c_var), 1))

            # if self.vae_train.args.discrete_action:
            #     discrete_action_eps = 1e-10
            #     discrete_action_eps_var = Variable(torch.Tensor(
            #         [discrete_action_eps])).type(self.dtype)
            #     # action_probs is (N, A)
            #     action_means = action_means + discrete_action_eps
            #     action_softmax = F.softmax(action_means, dim=1)
            #     action_probs = (action_var * action_softmax).sum(dim=1)

            #     action_means_old = action_means_old + discrete_action_eps
            #     action_old_softmax = F.softmax(action_means_old, dim=1)
            #     action_old_probs = (action_var * action_old_softmax).sum(dim=1)

            #     log_prob_cur = action_probs.log()
            #     log_prob_old = action_old_probs.log()
            # else:
            #     log_prob_cur = normal_log_density(action_var,
            #                                       action_means,
            #                                       action_log_stds,
            #                                       action_stds)

            #     log_prob_old = normal_log_density(action_var,
            #                                       action_means_old,
            #                                       action_log_stds_old,
            #                                       action_stds_old)

            # ==== Update value net ====
            if args.use_value_net:
                self.opt_value.zero_grad()
                value_inp_var = None
                # if self.args.use_goal_in_value:
                #     value_inp_var = torch.cat((state_var, goal_var), 1)
                # else: # change value net
                value_inp_var = torch.cat((state_var, latent_next_c_var), 1)
                value_var = self.value_net(value_inp_var)
                value_loss = (value_var - \
                        targets[curr_id:curr_id+curr_batch_size]).pow(2.).mean()
                value_loss.backward()
                self.opt_value.step()
                self.logger.summary_writer.add_scalar(
                        'loss/value',
                        value_loss.data.cpu().numpy(),
                        self.gail_step_count)
            # ==== END ====

            # ==== Update policy net (PPO step) ====
            # self.opt_policy.zero_grad()
            # ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            # surr1 = ratio * advantages_var[:, 0]
            # surr2 = torch.clamp(
            #         ratio,
            #         1.0 - self.args.clip_epsilon,
            #         1.0 + self.args.clip_epsilon) * advantages_var[:,0]
            # policy_surr = -torch.min(surr1, surr2).mean()
            # policy_surr.backward()
            
            # # torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 40)
            # self.opt_policy.step()
            # ==== END ====

            # compute old and new action probabilities
            policy_inp_var = torch.cat((state_var, latent_next_c_var), dim=1)
            _, policy_action_var = torch.max(action_var, dim=1)
            log_prob_cur = self.policy_net.get_log_prob(policy_inp_var,
                                                        policy_action_var)
            log_prob_old = self.old_policy_net.get_log_prob(policy_inp_var,
                                                            policy_action_var)

            # ==== Update policy net (PPO step) ====
            self.opt_policy.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var[:, 0]
            surr2 = torch.clamp(
                    ratio,
                    1.0 - self.args.clip_epsilon,
                    1.0 + self.args.clip_epsilon) * advantages_var[:,0]
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            # torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 40)
            self.opt_policy.step()
            # ==== END ====

            self.logger.summary_writer.add_scalar('loss/policy',
                                                  policy_surr.data.item(),
                                                  self.gail_step_count)

            policy_l2_norm, policy_grad_l2_norm = \
                              get_weight_norm_for_network(self.policy_net)
            self.logger.summary_writer.add_scalar('weight/policy/param',
                                                  policy_l2_norm,
                                                  self.gail_step_count)
            self.logger.summary_writer.add_scalar('weight/policy/grad',
                                                  policy_grad_l2_norm,
                                                  self.gail_step_count)

            # set new starting point for batch
            curr_id += curr_batch_size
            curr_id_exp += curr_batch_size_exp

            self.gail_step_count += 1
    def update_params(self, gen_batch, expert_batch, episode_idx,
                      optim_epochs, optim_batch_size):
        '''Update params for Policy (G), Reward (D) and Posterior (q) networks.
        '''
        args, dtype = self.args, self.dtype

        # generated trajectories
        states = torch.Tensor(np.array(gen_batch.state)).type(dtype)
        actions = torch.Tensor(np.array(gen_batch.action)).type(dtype)
        rewards = torch.Tensor(np.array(gen_batch.reward)).type(dtype)
        masks = torch.Tensor(np.array(gen_batch.mask)).type(dtype)
        goal = torch.Tensor(np.array(gen_batch.goal)).type(dtype)

        ## Expand states to include history ##
        # Generated trajectories already have history in them.

        latent_c = torch.Tensor(np.array(gen_batch.c)).type(dtype)
        latent_next_c = torch.Tensor(np.array(gen_batch.next_c)).type(dtype)
        values = None
        if args.use_value_net:
            value_net_inp = None
            # if self.args.use_goal_in_value:
            #     value_net_inp = torch.cat((states, goal), 1)
            # else:
            value_net_inp = torch.cat((states, latent_next_c), 1)
            values = self.value_net(Variable(value_net_inp))

        # expert trajectories
        list_of_expert_states, list_of_expert_actions = [], []
        list_of_expert_latent_c, list_of_masks = [], []
        list_of_expert_goals = []
        for i in range(len(expert_batch.state)):
            # c sampled from expert trajectories is incorrect since we don't
            # have "true c". Hence, we use the trained VAE to get the "true c".
            expert_c, _ = self.get_c_for_traj(
                    expert_batch.state[i][np.newaxis, :],
                    expert_batch.action[i][np.newaxis, :],
                    expert_batch.c[i][np.newaxis, :])

            # Remove b
            # expert_c[0, :] is c_{-1} which does not map to s_0. Hence drop it.
            expert_c = expert_c.squeeze(0)[1:, :]

            expert_goal = None
            if self.vae_train.use_rnn_goal_predictor:
                raise ValueError("Not implemented.")
            else:
                if self.args.env_type == 'grid_room':
                    expert_goal = expert_batch.c[i]
                else:
                    raise ValueError("Not implemented.")

            ## Expand expert states ##
            expanded_states = self.expand_states_numpy(expert_batch.state[i],
                                                       self.history_size)
            list_of_expert_states.append(torch.Tensor(expanded_states))
            list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
            list_of_expert_latent_c.append(torch.Tensor(expert_c))
            list_of_expert_goals.append(torch.Tensor(expert_goal))
            list_of_masks.append(torch.Tensor(expert_batch.mask[i]))

        expert_states = torch.cat(list_of_expert_states,0).type(dtype)
        expert_actions = torch.cat(list_of_expert_actions, 0).type(dtype)
        expert_latent_c = torch.cat(list_of_expert_latent_c, 0).type(dtype)
        expert_goals = torch.cat(list_of_expert_goals, 0).type(dtype)
        expert_masks = torch.cat(list_of_masks, 0).type(dtype)

        assert expert_states.size(0) == expert_actions.size(0), \
                "Expert transition size do not match"
        assert expert_states.size(0) == expert_latent_c.size(0), \
                "Expert transition size do not match"
        assert expert_states.size(0) == expert_masks.size(0), \
                "Expert transition size do not match"

        # compute advantages
        returns, advantages = get_advantage_for_rewards(rewards,
                                                        masks,
                                                        self.args.gamma,
                                                        self.args.tau,
                                                        values=values,
                                                        dtype=dtype)
        targets = Variable(returns)
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Backup params after computing probs but before updating new params
        for old_policy_param, policy_param in zip(self.old_policy_net.parameters(),
                                                  self.policy_net.parameters()):
            old_policy_param.data.copy_(policy_param.data)

        # update value, reward and policy networks
        optim_iters = self.args.batch_size // optim_batch_size
        optim_batch_size_exp = expert_actions.size(0) // optim_iters

        # Remove extra 1 array shape from actions, since actions were added as
        # 1-hot vector of shape (1, A).
        actions = np.squeeze(actions)
        expert_actions = np.squeeze(expert_actions)

        for _ in range(optim_epochs):
            perm = np.random.permutation(np.arange(actions.size(0)))
            perm_exp = np.random.permutation(np.arange(expert_actions.size(0)))
            if args.cuda:
                perm = torch.cuda.LongTensor(perm)
                perm_exp = torch.cuda.LongTensor(perm_exp)
            else:
                perm = torch.LongTensor(perm)
                perm_exp = torch.LongTensor(perm_exp)

            self.update_params_for_batch(
                states[perm],
                actions[perm],
                latent_c[perm],
                latent_next_c[perm],
                targets[perm],
                advantages[perm],
                expert_states[perm_exp],
                expert_actions[perm_exp],
                expert_latent_c[perm_exp],
                optim_batch_size,
                optim_batch_size_exp,
                optim_iters,
                goal=goal[perm],
                expert_goal=expert_goals[perm_exp])

    def train_gail(self, num_epochs, results_pkl_path,
                   gen_batch_size=1, train=True):
        '''Train GAIL.'''
        args, dtype = self.args, self.dtype
        results = {'average_reward': [], 'episode_reward': [],
                   'true_traj_state': {}, 'true_traj_action': {},
                   'pred_traj_state': {}, 'pred_traj_action': {}}

        self.train_step_count, self.gail_step_count = 0, 0
        gen_traj_step_count = 0
        self.convert_models_to_type(dtype)

        for ep_idx in trange(num_epochs):
            memory = Memory()

            num_steps, batch_size = 0, 1
            reward_batch, expert_true_reward_batch = [], []
            true_traj_curr_epoch = {'state':[], 'action': []}
            gen_traj_curr_epoch = {'state': [], 'action': []}
            env_reward_batch_dict = {'linear_traj_reward': [],
                                     'map_traj_reward': []}
            if args.history_size > 1:
                x_hist = -1 * np.ones(
                        (1, args.history_size, args.state_size),
                            dtype=np.float32)
            while num_steps < gen_batch_size: # batch=2048
                traj_expert = self.expert.sample(size=batch_size)
                state_expert, action_expert, c_expert, _ = traj_expert
                state_expert = np.array(state_expert, dtype=np.float32)
                action_expert = np.array(action_expert, dtype=np.float32)
                c_expert = np.array(c_expert, dtype=np.float32)
                # Generate c from trained VAE
                c_gen, expert_goal = self.get_c_for_traj(state_expert,
                                                         action_expert,
                                                         c_expert)

                expert_episode_len = state_expert.shape[1]

                # ==== Env reward for debugging (Grid envs only) ====
                # Create state expert map for reward
                state_action_expert_dict = {}
                for t in range(expert_episode_len):
                    pos_tuple = tuple(state_expert[0, t, :].astype(
                        np.int32).tolist())
                    state_action_key = (pos_tuple,
                                        np.argmax(action_expert[0, t, :]))
                    state_action_expert_dict[state_action_key] = 1
                # ==== END  ====

                if self.args.env_type == 'grid_room':
                    true_goal_numpy = np.copy(c_expert)
                else:
                    true_goal_numpy = np.zeros((c_expert.shape[0],
                                                self.num_goals))
                    true_goal_numpy[np.arange(c_expert.shape[0]),
                                    c_expert[:, 0]] = 1

                true_goal = Variable(torch.from_numpy(true_goal_numpy)).type(
                            self.dtype)

                # Sample start state or should we just choose the start state
                # from the expert trajectory sampled above.
                x_state_obj = StateVector(state_expert[:, 0, :], self.obstacles)
                x_feat = self.get_state_features(x_state_obj,
                                                 self.args.use_state_features)
                x = x_feat

                # Add history to state
                if args.history_size > 1:
                    # x_hist = -1 * np.ones(
                    #         (x.shape[0], args.history_size, x.shape[1]),
                    #         dtype=np.float32)
                    pdb.set_trace()
                    x_hist[:, :-1, :] = x_hist[:, 1:, :]
                    x_hist[:, (args.history_size-1), :] = x_feat
                    # x = self.get_history_features(x_hist)
                    x = x_hist

                # TODO: Make this a separate function. Can be parallelized.
                ep_reward, expert_true_reward = 0, 0
                env_reward_dict = {'linear_traj_reward': 0.0,
                                    'map_traj_reward': 0.0,}
                disc_reward, posterior_reward = 0.0, 0.0
                # Use a hard-coded list for memory to gather experience since
                # we need to mutate it before finally creating a memory object.
                memory_list = []
                curr_state_arr = state_expert[:, 0, :]
                
                for t in range(expert_episode_len):
                    # get c
                    ct, next_ct = c_gen[:, t, :], c_gen[:, t+1, :]
                    # ==== Get variables ====
                    # Get state and context variables
                    x_var = Variable(torch.from_numpy(
                        x.reshape((batch_size, -1))).type(self.dtype))
                    c_var = Variable(torch.from_numpy(
                        ct.reshape((batch_size, -1))).type(self.dtype))
                    next_c_var = Variable(torch.from_numpy(
                        next_ct.reshape((batch_size, -1))).type(self.dtype))

                    # Get the goal variable (either true or predicted)
                    goal_var = None
                    if self.vae_train.args.use_rnn_goal:
                        raise ValueError("To be implemented.")
                    else:
                        if len(true_goal.size()) == 2:
                            goal_var = true_goal
                        elif len(true_goal.size()) == 3:
                            goal_var = true_goal[:, t, :]
                        else:
                            raise ValueError("incorrect true goal size")
                    # ==== END ====
                    
                    # pdb.set_trace()
                    action = self.select_action(x_var, c_var).data.cpu().numpy()

                    # Take epsilon-greedy action only during training.
                    eps_low, eps_high = 0.1, 0.9
                    if not train:
                        eps_low, eps_high = 0.0, 0.0
                    action = epsilon_greedy_linear_decay(
                            action,
                            args.num_epochs * 0.5,
                            ep_idx,
                            self.action_size,
                            low=eps_low,
                            high=eps_high)

                    # Get the discriminator reward
                    disc_reward_t = self.get_discriminator_reward(x_var,
                                                                  action,
                                                                  c_var)
                    disc_reward += disc_reward_t

                    # Posterior reward
                    # posterior_reward_t = 0

                    # Update Rewards
                    ep_reward += (disc_reward_t)

                    # Since grid world environments don't have a "true" reward
                    # let us fake the true reward.
                    if self.args.env_type == 'grid_room':
                        curr_position = curr_state_arr.reshape(-1).astype(
                                np.int32).tolist()
                        expert_position = state_expert[0, t, :].astype(
                                np.int32).tolist()
                        if curr_position == expert_position:
                            env_reward_dict['linear_traj_reward'] += 1.0
                        expert_true_reward += 1.0

                        # Map reward. Each state should only be counted once
                        # only.
                        gen_state_action_key = (tuple(curr_position), action)
                        if state_action_expert_dict.get(
                                gen_state_action_key) is not None:
                            env_reward_dict['map_traj_reward'] += 1.0
                            del state_action_expert_dict[gen_state_action_key]
                    else:
                        pass
                        '''
                        true_goal_state = [int(x) for x in state_expert[-1].tolist()]
                        if self.args.flag_true_reward == 'grid_reward':
                          ep_true_reward += self.true_reward.reward_at_location(
                              curr_state_obj.coordinates, goals=[true_goal_state])
                          expert_true_reward += self.true_reward.reward_at_location(
                                state_expert[t], goals=[true_goal_state])
                        elif self.args.flag_true_reward == 'action_reward':
                          ep_true_reward += self.true_reward.reward_at_location(
                              np.argmax(action_expert[t]), action)
                          expert_true_reward += self.true_reward.corret_action_reward
                        else:
                          raise ValueError("Incorrect true reward type")
                        '''


                    # ==== Update next state =====
                    action_vec = ActionVector(np.array([action]))
                    # Get current state
                    state_vec = StateVector(curr_state_arr, self.obstacles)
                    # Get next state
                    next_state = self.transition_func(
                        state_vec, action_vec, 0)
                    next_state_feat = self.get_state_features(
                            next_state, self.args.use_state_features)
                    # ==== END ====


                    #next_state = running_state(next_state)
                    mask = 0 if t == expert_episode_len - 1 else 1

                    # ==== Push to memory ====
                    memory_list.append([
                        x.copy().reshape(-1),
                        np.array([oned_to_onehot(action, self.action_size)]),
                        mask,
                        next_state.coordinates,
                        disc_reward_t,
                        ct.reshape(-1),
                        next_ct.reshape(-1),
                        np.copy(goal_var.data.cpu().numpy()).reshape(-1)]) # goal_var, but shouldn't use
                    # ==== END ====

                    # ==== Set curr_state = next_state ====
                    if self.history_size > 1:
                        x_hist[:, :-1, :] = x_hist[:, 1:, :]
                        x_hist[:, self.args.history_size-1] = next_state_feat
                        x = self.get_history_features(x_hist)
                    else:
                        x[:] = next_state_feat
                    # ==== END ====

                    if args.render:
                        env.render()

                    if mask == 0:
                        break

                    # Update current state
                    curr_state_arr = np.array(next_state.coordinates,
                                              dtype=np.float32)


                assert memory_list[-1][2] == 0, \
                        "Mask for final end state is not 0."
                # Useless loop - remains here since we converted
                for memory_t in memory_list:
                    memory.push(*memory_t)

                if train:
                    reward_dict_stats = {
                        'discriminator': disc_reward,
                        'discriminator_per_step': disc_reward/expert_episode_len,
                    }
                    add_scalars_to_summary_writer(
                        self.logger.summary_writer,
                        'gen_traj/gen_reward',
                        reward_dict_stats,
                        gen_traj_step_count,
                    )

                num_steps += expert_episode_len

                # ==== Log rewards ====
                reward_batch.append(ep_reward)
                env_reward_batch_dict['linear_traj_reward'].append(
                        env_reward_dict['linear_traj_reward'])
                env_reward_batch_dict['map_traj_reward'].append(
                        env_reward_dict['map_traj_reward'])

                expert_true_reward_batch.append(expert_true_reward)
                results['episode_reward'].append(ep_reward)
                # ==== END ====

                # Increment generated trajectory step count.
                gen_traj_step_count += 1

            results['average_reward'].append(np.mean(reward_batch))

            # Add to tensorboard if training.
            linear_traj_reward = env_reward_batch_dict['linear_traj_reward']
            map_traj_reward = env_reward_batch_dict['map_traj_reward']
            if train:
                add_scalars_to_summary_writer(
                        self.logger.summary_writer,
                        'gen_traj/reward', {
                            'average': np.mean(reward_batch),
                            'max': np.max(reward_batch),
                            'min': np.min(reward_batch)
                        },
                        self.train_step_count)
                add_scalars_to_summary_writer(
                        self.logger.summary_writer,
                        'gen_traj/true_reward', {
                            'average': np.mean(linear_traj_reward),
                            'max': np.max(linear_traj_reward),
                            'min': np.min(linear_traj_reward),
                            'expert_true': np.mean(expert_true_reward_batch),
                            'map_average': np.mean(map_traj_reward),
                            'map_max': np.max(map_traj_reward),
                            'map_min': np.min(map_traj_reward),
                        },
                        self.train_step_count)

            if train:
                # ==== Update parameters ====
                gen_batch = memory.sample()

                # We do not get the context variable from expert trajectories.
                # Hence we need to fill it in later.
                expert_batch = self.expert.sample(size=args.num_expert_trajs)

                self.update_params(gen_batch, expert_batch, ep_idx,
                                   args.optim_epochs, args.optim_batch_size)

                self.train_step_count += 1

            if ep_idx > 0 and  ep_idx % args.log_interval == 0:
                print('Episode [{}/{}]  Avg R: {:.2f}   Max R: {:.2f} \t' \
                      'True Avg {:.2f}   True Max R: {:.2f}   ' \
                      'Expert (Avg): {:.2f}   ' \
                      'Dict(Avg): {:.2f}    Dict(Max): {:.2f}'.format(
                      ep_idx, args.num_epochs, np.mean(reward_batch),
                      np.max(reward_batch), np.mean(linear_traj_reward),
                      np.max(linear_traj_reward),
                      np.mean(expert_true_reward_batch),
                      np.mean(map_traj_reward),
                      np.max(map_traj_reward)))

            with open(results_pkl_path, 'wb') as results_f:
                pickle.dump((results), results_f, protocol=2)

            if train and ep_idx > 0 and ep_idx % args.save_interval == 0:
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(torch.FloatTensor)
                checkpoint_filepath = self.model_checkpoint_filepath(ep_idx)
                torch.save(self.checkpoint_data_to_save(), checkpoint_filepath)
                print("Did save checkpoint: {}".format(checkpoint_filepath))
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(self.dtype)

