import wandb
from bcolors import bcolors
from boundary_utils import get_boundary_from_all_traj
from re import L
from tqdm import trange
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
from grid_world import create_obstacles
from plot_utils import plot_pickle_results_context

import grid_world as gw
from load_expert_traj import Expert, ExpertHDF5, CircleExpertHDF5
# from load_expert_traj import recursively_save_dict_contents_to_group
# from itertools import product

from utils.logger import Logger, TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network
from base_vae import DiscreteVAE, VAE
global_args = None
from itertools import product

class VAETrain(object):
    def __init__(self, args,
                 logger,
                 width=11,
                 height=15,
                 state_size=2,
                 action_size=4,
                 dtype=torch.FloatTensor,
                 env_type='grid',
                 env_name=None, use_boundary=False):
        self.obstacles, self.rooms, _ = create_obstacles(width, height, env_name='room',
                                    room_size=3)
        self.set_diff = list(set(product(tuple(range(0, width)),tuple(range(0, height)))) \
                - set(self.obstacles))
        self.args = args
        self.logger = logger
        self.width, self.height = width, height
        self.state_size = state_size
        self.action_size = action_size
        self.dtype = dtype
        self.env_type = env_type
        self.env_name = env_name
        self.use_boundary = use_boundary

        self.train_step_count = 0
        self.device = 'cuda' if args.cuda else 'cpu'
        # Create models
        # self.Q_model = nn.LSTMCell(self.state_size + action_size, 64)
        # self.Q_2_model = nn.LSTMCell(64, 64)

        # Output of linear model num_goals = 4
        # self.Q_model_linear = nn.Linear(64, num_goals)

        # action_size is 0
        # Hack -- VAE input dim (s + a + latent).
        self.vae_model = DiscreteVAE(
                temperature=args.temperature,
                policy_state_size=state_size,
                posterior_state_size=state_size,
                policy_latent_size=args.vae_context_size,
                posterior_latent_size=args.vae_context_size,
                policy_output_size=action_size,
                hidden_size=64,
                use_boundary=args.use_boundary,
                dynamics_state_size=len(self.set_diff),
                args=args)

        self.obstacles, self.transition_func = None, None

        if args.run_mode == 'train':
            wd = 1e-5
            self.policy_opt = optim.Adam(self.vae_model.policy.parameters(), lr=args.policy_lr, weight_decay=wd)
            self.posterior_opt = optim.Adam(self.vae_model.posterior.parameters(), lr=args.posterior_lr, weight_decay=wd)
            self.transition_opt = optim.Adam(self.vae_model.transition.parameters(), lr=args.transition_lr, weight_decay=wd)
            self.Q_model_opt = optim.Adam([
                    {'params': self.vae_model.Q_alpha.parameters()},
                    {'params': self.vae_model.Q_gamma.parameters()},
                ],
                lr=args.dynamics_lr, weight_decay=wd)

        elif args.run_mode == 'test' or args.run_mode == 'test_goal_pred':
            pass
        else:
            raise ValueError('Incorrect value for run_mode {}'.format(
                args.run_mode))
        self.create_environment(env_type, env_name)
        self.expert = None
        # self.obstacles, self.set_diff = None, None
    
    def get_boundary(self):
        # get boundary
        if self.use_boundary:
            traj_expert = self.expert.sample_all()
            state_expert, action_expert, c_expert, _ = traj_expert
            self.boundary_list = get_boundary_from_all_traj(state_expert)     

    def model_checkpoint_dir(self):
        '''Return the directory to save models in.'''
        return os.path.join(self.args.results_dir, 'checkpoint')

    def model_checkpoint_filename(self, epoch):
        return os.path.join(self.model_checkpoint_dir(),
                            'cp_{}.pth'.format(epoch))

    def create_environment(self, env_type, env_name=None):
        if 'grid' in env_type:
            self.transition_func = gw.TransitionFunction(self.width,
                                                         self.height,
                                                         gw.obstacle_movement)
        else:
            raise ValueError("Invalid env type: {}".format(env_type))

    def set_expert(self, expert):
        assert self.expert is None, "Trying to set non-None expert"
        self.expert = expert
        assert expert.obstacles is not None, "Obstacles cannot be None"
        self.obstacles = expert.obstacles
        assert expert.set_diff is not None, "set_diff in h5 file cannot be None"
        self.set_diff = expert.set_diff

    def convert_models_to_type(self, dtype):
        self.vae_model = self.vae_model.type(dtype)

    def KLD_loss(self, logits):
        q_prob = F.softmax(logits, dim=1)  # q_prob
        log_q_prob = torch.log(q_prob + 1e-10)  # log q_prob
        prior_prob = Variable(torch.Tensor([1.0 / self.vae_model.posterior_latent_size])).type(
                logits.data.type())
        batch_size = logits.size(0)
        KLD = torch.sum(q_prob * (log_q_prob - torch.log(prior_prob))) / batch_size
        return KLD

    def log_model_to_tensorboard(
            self,
            models_to_log=('policy', 'posterior', 'transition', 'Q_alpha', 'Q_gamma')):
        '''Log weights and gradients of network to Tensorboard.

        models_to_log: Tuple of model names to log to tensorboard.
        '''
        for model_name in models_to_log:
            if model_name == 'policy':
                vae_model_l2_norm, vae_model_grad_l2_norm = get_weight_norm_for_network(self.vae_model.policy)
            if model_name == 'posterior':
                vae_model_l2_norm, vae_model_grad_l2_norm = get_weight_norm_for_network(self.vae_model.posterior)
            if model_name == 'transition':
                vae_model_l2_norm, vae_model_grad_l2_norm = get_weight_norm_for_network(self.vae_model.transition)
            if model_name == 'q_alpha':
                vae_model_l2_norm, vae_model_grad_l2_norm = get_weight_norm_for_network(self.vae_model.Q_alpha)
            if model_name == 'q_gamma':
                vae_model_l2_norm, vae_model_grad_l2_norm = get_weight_norm_for_network(self.vae_model.Q_gamma)
            
            if self.args.wandb:
                wandb.log({f'weight and gradients/{model_name} network/Weight':vae_model_l2_norm})
                wandb.log({f'weight and gradients/{model_name} network/Grads':vae_model_grad_l2_norm})
            # self.logger.summary_writer.add_scalar(
            #                 'weight/policy',
            #                  vae_model_l2_norm,
            #                  self.train_step_count)
            # self.logger.summary_writer.add_scalar(
            #                 'grad/policy',
            #                  vae_model_grad_l2_norm,
            #                  self.train_step_count)

        # if 'vae_posterior' in models_to_log:
        #     vae_model_l2_norm, vae_model_grad_l2_norm = \
        #                     get_weight_norm_for_network(self.vae_model.posterior)
        #     if self.args.wandb:
        #         wandb.log({'weight and gradients/posterior network/Weight':vae_model_l2_norm})
        #         wandb.log({'weight and gradients/posterior network/Grads':vae_model_grad_l2_norm})

            # self.logger.summary_writer.add_scalar(
            #                 'weight/posterior',
            #                  vae_model_l2_norm,
            #                  self.train_step_count)
            # self.logger.summary_writer.add_scalar(
            #                 'grad/posterior',
            #                  vae_model_grad_l2_norm,
            #                  self.train_step_count)



    def set_models_to_train(self):
        self.vae_model.train()

    def save_checkpoint(self, epoch):
        model_data = {
                'vae_model': self.vae_model.state_dict(),
        }
        torch.save(model_data, self.model_checkpoint_filename(epoch))

    def load_checkpoint(self, checkpoint_path):
        '''Load models from checkpoint.'''
        checkpoint_models = torch.load(checkpoint_path)
        self.vae_model.load_state_dict(checkpoint_models['vae_model'])

    def load_checkpoint_goal_policy(self, checkpoint_path):
        '''Load models from checkpoint.'''
        checkpoint_models = torch.load(checkpoint_path)
        self.vae_model.load_state_dict(checkpoint_models['vae_model'])

    def get_state_features(self, state_obj, use_state_features):
        if use_state_features:
            feat = np.array(state_obj.get_features(), dtype=np.float32)
        else:
            feat = np.array(state_obj.coordinates, dtype=np.float32)
        return feat


    def get_context_at_state(self, x, c):
        '''Get context variable c_t for given x_t, c_{t-1}.

        x: State at time t. (x_t)
        c: Context at time t-1. (c_{t-1})
        '''
        if self.args.use_discrete_vae:
            logits = self.vae_model.encode(x, c)
            return self.vae_model.reparameterize(logits,
                                                 self.vae_model.temperature)
        else:
            mu, logvar = self.vae_model.encode(x, c)
            return self.vae_model.reparameterize(mu, logvar)

    def train(self, expert, num_epochs, batch_size):
        final_train_stats = {
            'train_loss': [],
            'goal_pred_conf_arr': [],
            'temperature': [],
        }
        self.train_step_count = 0
        # Convert models to right type.
        self.convert_models_to_type(self.dtype)

        # Create the checkpoint directory.
        if not os.path.exists(self.model_checkpoint_dir()):
            os.makedirs(self.model_checkpoint_dir())
        # Save runtime arguments to pickle file
        args_pkl_filepath = os.path.join(self.args.results_dir, 'args.pkl')
        with open(args_pkl_filepath, 'wb') as args_pkl_f:
            pickle.dump(self.args, args_pkl_f, protocol=2)

        pdb.set_trace()
        for epoch in trange(1, num_epochs+1):
            self.vae_model.update_temperature(epoch-1)
            train_stats = self.train_varied_length_epoch(epoch, expert, batch_size)

            # Update stats for epoch
            final_train_stats['train_loss'].append(train_stats['train_loss'])
            if self.args.use_discrete_vae:
                final_train_stats['temperature'].append(
                        self.vae_model.temperature)

            if epoch % 25 == 0 or epoch == 1:
            # if epoch % 1000 == 0:
                results_pkl_path = os.path.join(self.args.results_dir, f'pred_result_cp_{epoch}.pkl')
                self.test_models(expert, results_pkl_path=results_pkl_path,
                                 num_test_samples=20)
                plot_pickle_results_context(results_pkl_path=results_pkl_path, obstacles=self.obstacles, rooms=self.rooms, num_traj_to_plot=20)

            if epoch % self.args.checkpoint_every_epoch == 0:
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(torch.FloatTensor)
                self.save_checkpoint(epoch)
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(self.dtype)

        results_pkl_path = os.path.join(self.args.results_dir, 'results.pkl')
        self.test_models(expert, results_pkl_path=results_pkl_path,
                         num_test_samples=5,
                         other_results_dict={'train_stats': final_train_stats})


    def train_varied_length_epoch(self, epoch, expert, batch_size=1):
        '''Train VAE with fixed length expert samples.
        '''
        self.set_models_to_train()
        train_stats = {
            'train_loss': [],
        }

        # TODO: The current sampling process can retrain on a single trajectory
        # multiple times. Will fix it later.
        num_batches = expert.memory_count // batch_size + 1
        total_epoch_loss, total_epoch_per_step_loss = 0.0, 0.0

        for batch_idx in range(num_batches):
            total_pair_count = 0
            batch_posterior_loss = []
            batch_policy_loss = []
            batch_transition_loss = []
            batch_dynamics_loss_sum = []
            batch_dynamics_loss = []
            self.policy_opt.zero_grad()
            self.posterior_opt.zero_grad()
            self.transition_opt.zero_grad()
            self.Q_model_opt.zero_grad()
            # keep rolling out until batch_size
            logits_q = 0.
            while total_pair_count < batch_size:
            # for batch_idx in range(num_batches): # 300 // 128 = 2
                # Train loss for this batch
                train_loss, train_policy_loss = 0.0, 0.0
                train_KLD_loss, train_posterior_loss, train_transition_loss = 0.0, 0.0, 0.0
                train_dynamics_loss = 0.0
                train_dynamics_loss_sum = 0.0
                ep_timesteps = 0
                batch = expert.sample(1)

                ep_state, ep_action, _, ep_mask = batch
                ep_state = torch.tensor(ep_state).type(self.dtype)
                ep_action = torch.tensor(ep_action).type(self.dtype) # (B, L, D)
                # action_var = Variable(torch.from_numpy(ep_action).type(self.dtype))
                total_pair_count += ep_state.shape[-2]

                # ep_state = torch.from_numpy(ep_state)
                boundary_time_stamp_list = []
                episode_len = ep_state.shape[1]
                # pdb.set_trace()
                for history_t in range(episode_len):
                    xy_posi = (ep_state[0][history_t][0], ep_state[0][history_t][1])
                    if xy_posi in self.boundary_list:
                        # print('boundary: ', xy_posi)
                        if history_t not in boundary_time_stamp_list:
                            boundary_time_stamp_list.append(history_t)
                        # (len(boundary_time_stamp_list) + 1, 10)
                # add 2 c_traj_var for begining and end
                c_traj_var = [F.softmax(torch.ones((ep_state.shape[0], self.vae_model.posterior_latent_size)).type(self.dtype), dim=-1) for _ in range(len(boundary_time_stamp_list) + 2)]

                # TODO all c shift for 1 timestep
                # c_var_pre = Variable(c_traj_var[:, 0, :]).type(self.dtype)
                sub_traj_index = 0
                cross_point_index = 0
                for t in range(episode_len):
                    # get history
                    ep_timesteps += 1

                    # \tau = x_from_last_boundary : current_t
                    x_var = ep_state[:, cross_point_index:t+1, :]
                    # x_var = Variable(torch.from_numpy(ep_state[:, cross_point_index:t+1, :]).type(self.dtype))
                    
                    # [1]. MI variational bound for posterior 
                    
                    # sample c_j from T( | c_{j-1}) and save in var_list 
                    if t in boundary_time_stamp_list or t == 0:
                        # update c and switch subtask
                        sub_traj_index += 1
                        # c_traj_var[sub_traj_index] = self.vae_model.reparameterize(logits=logits, temperature=self.vae_model.temperature).clone()
                        c_var_transition_to_current = F.softmax(self.vae_model.transition(c_traj_var[sub_traj_index - 1]), dim=-1) # (1, 10)
                        c_traj_var[sub_traj_index] = c_var_transition_to_current
                        if sub_traj_index > 1: # only when sub_traj_index ==2, begin to change sub_traj_id, skip the begining
                            cross_point_index = boundary_time_stamp_list[sub_traj_index - 2] + 1
                            if self.args.debug:
                                print(x_var)
                                print("Train set boundary to:", cross_point_index)
                    
                    # get posterior c' = Q(x_{1:t}, c-1)
                    logits_q = F.softmax(self.vae_model.posterior(x_var, c_traj_var[sub_traj_index - 1]), dim=-1) #(B, 10)
                    # reweight
                    weight = (1. / self.args.vae_context_size) / logits_q.detach()
                    # _, label = torch.max(c_traj_var[sub_traj_index].clone(), 1) #TODO
                    # posterior_loss = F.cross_entropy(logits_q, label.detach(), weight=weight[0])
                    posterior_loss = -(c_traj_var[sub_traj_index].clone() * torch.log(logits_q)).mean() 
                    KLD_loss = self.KLD_loss(logits_q)    
                    batch_posterior_loss.append(posterior_loss * self.args.lambda_posterior + KLD_loss * self.args.lambda_kld)
                    train_posterior_loss += posterior_loss.data.item()
                    train_KLD_loss += KLD_loss.data.item()
                    train_loss += batch_posterior_loss[-1].data.item()

                    if epoch > self.args.warmup_epochs and t < ep_state.shape[-2] - 1:
                        # [2] minMI
                        # log(s' | s, a, c, c_prev) - log(s' | s, a, c)
                        # TODO should this update transition? I think yes
                        # pdb.set_trace()

                        state_label = torch.tensor(self.set_diff.index(ep_state[0, t + 1, :].type(torch.int64).tolist())).unsqueeze(0).to(self.device)
                        dynamics_loss = F.cross_entropy(self.vae_model.Q_alpha(torch.cat((ep_state[:, t, :], ep_action[:, t, :], c_traj_var[sub_traj_index], c_traj_var[sub_traj_index - 1]), dim=-1)), state_label) - F.cross_entropy(self.vae_model.Q_gamma(torch.cat((ep_state[:, t, :], ep_action[:, t, :], c_traj_var[sub_traj_index]), dim=-1)), state_label)
                        batch_dynamics_loss.append(dynamics_loss)
                        train_dynamics_loss += dynamics_loss.data.item()
                        train_loss += dynamics_loss.data.item()
                    if t < ep_state.shape[-2] - 1:
                        state_label = torch.tensor(self.set_diff.index(ep_state[0, t + 1, :].type(torch.int64).tolist())).unsqueeze(0).to(self.device)
                        dynamics_loss_alpha = F.cross_entropy(self.vae_model.Q_alpha(torch.cat((ep_state[:, t, :], ep_action[:, t, :], c_traj_var[sub_traj_index], c_traj_var[sub_traj_index - 1]), dim=-1)), state_label) 
                        dynamics_loss_gamma = F.cross_entropy(self.vae_model.Q_gamma(torch.cat((ep_state[:, t, :], ep_action[:, t, :], c_traj_var[sub_traj_index]), dim=-1)), state_label)
                        batch_dynamics_loss_sum.append(dynamics_loss_alpha + dynamics_loss_gamma)
                        train_dynamics_loss_sum += batch_dynamics_loss_sum[-1].data.item()
                        train_loss += batch_dynamics_loss_sum[-1].data.item()
                    
                    #[3] policy loss #TODO use discreet q in policy?
                    action_mean = self.vae_model.policy(torch.cat((x_var[:, -1, :], logits_q.detach()), dim=-1)) #(B, 4), posterior (B, 10)
                    expert_action_var = ep_action[:, t, :]
                    _, label_a = torch.max(expert_action_var, 1)
                    policy_loss = F.cross_entropy(action_mean, label_a)
                    batch_policy_loss.append(policy_loss)
                    train_policy_loss += policy_loss.data.item()
            
                    #[4] Transition loss only for boundaries. take the output of posterior Q as c_{j-1}? #TODO
                    if t in boundary_time_stamp_list or t == 0: # calculate when t==0?
                        # calculate Transition loss
                        _, label_posterior = torch.max(logits_q.clone(), 1)
                        transition_loss = -(logits_q.clone() * torch.log(c_var_transition_to_current)).mean() 
                        # transition_loss = F.cross_entropy(c_var_transition_to_current, label_posterior.detach())
                        batch_transition_loss.append(transition_loss)
                        train_transition_loss += transition_loss.data.item()
                        train_loss += transition_loss.data.item()

            # Calculate the total loss.
            total_loss = batch_posterior_loss[0]
            for t in range(1, len(batch_posterior_loss)):
                total_loss = total_loss + batch_posterior_loss[t]
            # torch.mean(ep_posterior_loss).backward(retain_graph=True)
            # pdb.set_trace()
            (total_loss / len(batch_posterior_loss)).backward(retain_graph=True)
            
            total_loss_2 = batch_transition_loss[0]
            for t in range(1, len(batch_transition_loss)):
                total_loss_2 = total_loss_2 + batch_transition_loss[t]
            # total_loss.backward()
            (total_loss_2 / len(batch_transition_loss)).backward(retain_graph=True)

            total_loss = batch_policy_loss[0]
            for t in range(1, len(batch_policy_loss)):
                total_loss = total_loss + batch_policy_loss[t]
            (total_loss / len(batch_policy_loss)).backward(retain_graph=True)
            
            if epoch > self.args.warmup_epochs:
                total_loss = batch_dynamics_loss[0]
                for t in range(1, len(batch_dynamics_loss)):
                    total_loss = total_loss + batch_dynamics_loss[t]
                (total_loss / len(batch_dynamics_loss)).backward(retain_graph=True)
                # total_loss.backward()
                # pdb.set_trace()
                # self.transition_opt.step()
            else:
                total_loss = batch_dynamics_loss_sum[0]
                for t in range(1, len(batch_dynamics_loss_sum)):
                    total_loss = total_loss + batch_dynamics_loss_sum[t]
                (total_loss / len(batch_dynamics_loss_sum)).backward(retain_graph=True)
                
                
                if epoch % self.args.policy_interval == 0:
                    self.policy_opt.step()
                if epoch % self.args.posterior_interval == 0:
                    self.posterior_opt.step()
                if epoch % self.args.transition_interval == 0:
                    self.transition_opt.step()
                # if epoch % self.args.dynamics_interval == 0:
                if epoch < self.args.warmup_epochs:
                    self.Q_model_opt.step()

                # Get the gradients and network weights
                if self.args.log_gradients_tensorboard:
                    self.log_model_to_tensorboard()


            # Update stats
            total_epoch_loss += train_loss
            train_stats['train_loss'].append(train_loss)
            if self.args.wandb and batch_idx % self.args.log_interval == 0:
                loss_dict = {}
                loss_dict['training_loss/per_batch'] = train_loss
                loss_dict['training_loss/policy_loss_per_batch'] = train_policy_loss
                loss_dict['training_loss/KLD_loss_per_batch'] = train_KLD_loss
                loss_dict['training_loss/posterior_loss_per_batch'] = train_posterior_loss
                loss_dict['training_loss/transition_loss_per_batch'] = train_transition_loss
                loss_dict['training_loss/dynamics_loss_sum_per_batch'] = train_dynamics_loss_sum
                loss_dict['training_loss/dynamics_loss_per_batch'] = train_dynamics_loss
                wandb.log(loss_dict)
            # self.logger.summary_writer.add_scalar('training_loss/per_batch',
            #                                         train_loss,
            #                                         self.train_step_count)
            # self.logger.summary_writer.add_scalar(
            #         'training_loss/policy_loss_per_batch',
            #         train_policy_loss,
            #         self.train_step_count)
            # self.logger.summary_writer.add_scalar(
            #         'training_loss/KLD_loss_per_batch',
            #         train_KLD_loss,
            #         self.train_step_count)
            # self.logger.summary_writer.add_scalar(
            #         'training_loss/posterior_loss_per_batch',
            #         train_posterior_loss,
            #         self.train_step_count)
            # self.logger.summary_writer.add_scalar(
            #         'training_loss/transition_loss_per_batch',
            #         train_transition_loss,
            #         self.train_step_count)
            # self.logger.summary_writer.add_scalar(
            #         'training_loss/dynamics_loss_per_batch',
            #         train_dynamics_loss,
            #         self.train_step_count)
            

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{}] \t Loss: {:.3f} \t ' \
                        'Policy Loss: {:.2f}, \t Posterior Loss: {:.2f}, \t KLD: {:.2f}, \t Transition Loss: {:.2f}, \t Dynamics Loss:{:.2f}' \
                        .format(
                    epoch, batch_idx, num_batches, train_loss,
                    train_policy_loss, train_posterior_loss, train_KLD_loss, train_transition_loss, train_dynamics_loss))

            self.train_step_count += 1

        return train_stats

## ==============================================
## Test prediction

    def test_models(self, expert, results_pkl_path=None,
                    other_results_dict=None, num_test_samples=100):
        '''Test models by generating expert trajectories.'''
        self.convert_models_to_type(self.dtype)

        # if self.env_type == 'grid_room':
        #     results = self.test_generate_all_grid_states()
    # else:
        results = self.test_generate_trajectory_variable_length(
                expert,
                num_test_samples=num_test_samples)
        
        if other_results_dict is not None:
            # Copy other results dict into the main results
            for k, v in other_results_dict.items():
                results[k] = v

        # Save results in pickle file
        if results_pkl_path is not None:
            with open(results_pkl_path, 'wb') as results_f:
                pickle.dump(results, results_f, protocol=2)
                print('Did save results to {}'.format(results_pkl_path))

    def test_generate_trajectory_variable_length(self,
                                                 expert,
                                                 num_test_samples=10):
        '''Test trajectory generation from VAE.

        Use expert trajectories for trajectory generation.

        expert: Expert object.
        num_test_samples: Number of trajectories to sample.
            from goal policy.
        '''
        self.vae_model.eval()

        batch_size = 1

        results = {
                'true_traj_state': [], 'true_traj_action': [],
                'pred_traj_state': [], 'pred_traj_action': [],
                'pred_context': [], 'true_traj_pred_context': []}

        true_reward_batch = []

        # We need to sample expert trajectories to get (s, a) pairs which
        # are required for goal prediction.
        for e in range(num_test_samples):
            true_reward = 0.0
            batch = expert.sample_index(ind=e)

            ep_state, ep_action, _, ep_mask = batch
            ep_state = np.array(ep_state, dtype=np.float32)
            ep_action = np.array(ep_action, dtype=np.float32)# (B, L, D)
            # ep_action will be (N, A)
            action_var = Variable(
                    torch.from_numpy(ep_action).type(self.dtype))
            episode_len = ep_state.shape[1]
            boundary_time_stamp_list = []
            for history_t in range(episode_len):
                xy_posi = (ep_state[0][history_t][0], ep_state[0][history_t][1])
                if xy_posi in self.boundary_list:
                    # print('boundary: ', xy_posi)
                    if history_t not in boundary_time_stamp_list:
                        boundary_time_stamp_list.append(history_t)

            # c_traj =  -1 * np.ones((batch_size, ep_state.shape[1], self.vae_model.posterior_latent_size),
            #                  dtype=np.float32) # (B, L, 10)
            # c_traj_var = -1 * torch.ones((len(boundary_time_stamp_list) + 2, self.vae_model.posterior_latent_size),
            #                 dtype=torch.float32, requires_grad=True) # (B, count_L, 10)
            c_traj_var = [F.softmax(torch.ones((ep_state.shape[0], self.vae_model.posterior_latent_size)).type(self.dtype), dim=-1) for _ in range(len(boundary_time_stamp_list) + 2)]

            # Store list of losses to backprop later.
            # ep_loss, curr_state_arr = [], ep_state[:, 0, :]
            pred_context = []
            true_traj = []
            # pred_traj = []
            sub_traj_index = 0
            cross_point_index = 0
            for t in range(ep_state.shape[1]):

                x_var = Variable(torch.from_numpy(ep_state[:, cross_point_index:t+1, :]).type(self.dtype))

                # else: # add true goal after c, but didn't use.  # (B, history-1, 10)
                # TODO all c shift for 1 timestep
                
                # vae_output = self.vae_model(x_var, c_traj_var[:, sub_traj_index, :]) # output = (B, 4), posterior (B, 10)
                # logit = F.softmax(self.vae_model.posterior(x_var), dim=-1)
                expert_action_var = action_var[:, t, :].clone()
                if t in boundary_time_stamp_list or t == 0:
                    # update c and switch subtask
                    sub_traj_index += 1
                    # c_traj_var[sub_traj_index] = self.vae_model.reparameterize(logits=logits, temperature=self.vae_model.temperature).clone()
                    c_var_transition_to_current = F.softmax(self.vae_model.transition(c_traj_var[sub_traj_index - 1].type(self.dtype)), dim=-1) # (1, 10)
                    c_traj_var[sub_traj_index] = c_var_transition_to_current
                    if sub_traj_index > 1: # only when sub_traj_index ==2, begin to change sub_traj_id, skip the begining
                        cross_point_index = boundary_time_stamp_list[sub_traj_index - 2] + 1
                        if self.args.debug:
                            print(cross_point_index, 'is cross_point_index')
                    # get posterior c' = Q(x_{1:t}, c-1)
                # if t in boundary_time_stamp_list:
                #     # pdb.set_trace()
                #     cross_point_index = boundary_time_stamp_list[sub_traj_index] + 1
                #     sub_traj_index += 1
                #     c_traj_var[:, sub_traj_index, :] = self.vae_model.reparameterize(*vae_reparam_input)
                    # store latent variables (logits or mu)
                    if t != 0:
                        logits_q = F.softmax(self.vae_model.posterior(x_var, c_traj_var[sub_traj_index - 1].type(self.dtype)), dim=-1) #(B, 10)
                        for _ in range(x_var.shape[-2]):
                            pred_context.append(logits_q.data.cpu().numpy())
                        if self.args.debug:
                            print(x_var)
                            print(len(pred_context))

                # vae_reparam_input = (logit, self.vae_model.temperature)
            # pdb.set_trace()


                # pred_actions_numpy = vae_output[0].data.cpu().numpy()

                # Store the "true" state
                true_traj.append((ep_state[:, t, :], ep_action[:, t, :]))
                # pred_traj.append((curr_state_arr, pred_actions_numpy))


                # update c
                # c[:, -self.vae_model.posterior_latent_size:] = \
                #             self.vae_model.reparameterize(
                #                     *vae_reparam_input).data.cpu() # (B, 10)

            '''
            ===== Print predicted trajectories while debugging ====
            def get_traj_from_tuple(x):
                traj = []
                for i in x:
                    traj.append((i[0][0], i[0][1]))
                return traj

            print('True: {}'.format(get_traj_from_tuple(true_traj)))
            print('Pred: {}'.format(get_traj_from_tuple(pred_traj)))
            '''
            if len(pred_context) < ep_state.shape[-2]:
                # only add this when last_time_step is not a boundary
                # pdb.set_trace()
                for _ in range(x_var.shape[-2]):
                    pred_context.append(logits_q.data.cpu().numpy())
            true_traj_state_arr = np.array([x[0] for x in true_traj])
            # true_traj_action_arr = np.array([x[1] for x in true_traj])
            # pred_traj_state_arr = np.array([x[0] for x in pred_traj])
            # pred_traj_action_arr = np.array([x[1] for x in pred_traj])

            results['true_traj_state'].append(true_traj_state_arr)
            # results['true_traj_action'].append(true_traj_action_arr)
            # results['pred_traj_state'].append(pred_traj_state_arr)
            # results['pred_traj_action'].append(pred_traj_action_arr)
            results['pred_context'].append(np.array(pred_context))

            true_reward_batch.append(true_reward)

        true_reward_batch = np.array(true_reward_batch)
        print('Batch reward mean: {:.3f}'.format(true_reward_batch.mean()))
        print('Batch reward std: {:.3f}'.format(true_reward_batch.std()))

        return results


    def test_generate_all_grid_states(self):

        '''Test trajectory generation at each state for room_trajs.'''
        self.vae_model.eval()

        history_size, batch_size = self.vae_model.history_size, 1

        results = {'pred_traj_action': {}}

        # We need to sample expert trajectories to get (s, a) pairs which
        # are required for goal prediction.
        print('width:', self.width)
        print('hight:', self.height)
        # print('posterior_goal_size:', self.vae_model.posterior_goal_size)
        print('posterior_latent_size', self.vae_model.posterior_latent_size)
        list_of_obstacles = self.obstacles.tolist()
        # print(list_of_obstacles)
        for i in range(self.width):
            for j in range(self.height):
                if [i, j] in list_of_obstacles:
                    continue
                for g in range(self.vae_model.posterior_goal_size): # check num_goals
                    true_goal_numpy = np.zeros((1, self.vae_model.posterior_goal_size))
                    true_goal_numpy[0, g] = 1.0
                    true_goal = Variable(torch.from_numpy(true_goal_numpy).type(
                                         self.dtype))

                    for context in range(self.vae_model.posterior_latent_size):
                        # Get the initial state
                        c = np.zeros((1, self.vae_model.posterior_latent_size),
                                      dtype=np.float32)
                        c[0, context] = 1.0

                        x_state_obj = gw.StateVector(np.array([i, j])[np.newaxis, :], 
                                                  self.obstacles)
                        x_feat = self.get_state_features(x_state_obj,
                                                         self.args.use_state_features)


                        # x is (1, F)
                        x = x_feat

                        x_var = Variable(torch.from_numpy(
                                         x.reshape((batch_size, -1))).type(self.dtype))

                        c_var = Variable(torch.from_numpy(c).type(self.dtype))
                        c_var = torch.cat([true_goal, c_var], dim=1)

                        action = self.vae_model.decode(x_var[:, -self.vae_model.policy_state_size:],
                                                           c_var[:,-self.vae_model.posterior_latent_size:])

                        pred_actions_numpy = action.data.cpu().numpy()
                        
                        # ij is position
                        # context=4, g=4, 
                        #               where g doesn't influence action.
                        results['pred_traj_action'][(i, j, context, g)] = pred_actions_numpy
        return results

