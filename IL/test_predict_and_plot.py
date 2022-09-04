from load_expert_traj import Expert, ExpertHDF5, CircleExpertHDF5
import torch
from grid_world_gail import load_VAE_model
import argparse
import pickle
import os
parser = argparse.ArgumentParser('predict and plot context for expert trajectories')
# paths
parser.add_argument('--expert_path', default='./h5_trajs/room_trajs/traj_room_centre_len_50') # expert_path = './h5_trajs/room_trajs/traj_len_16'
parser.add_argument('--vae_checkpoint_path', default='./results/vae/room_traj/discrete/centre_only_temp_1_0.1_context_4/checkpoint/cp_1000.pth')
parser.add_argument('--results_pkl_path', default='./results/test_predict/room_traj/centre_only_temp_1_0.1_context_4/pred_result_cp_1000.pth')
parser.add_argument('--results_dir', default='./results/test_predict/room_traj/centre_only_temp_1_0.1_context_4')
# parameter
parser.add_argument('--vae_state_size', default=4, type=int)
# trainig phase
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--use_state_features', action='store_true')

args = parser.parse_args()


# load expert trajectories
expert = ExpertHDF5(args.expert_path, args.vae_state_size)
expert.push(only_coordinates_in_state=True, one_hot_action=True)

# load vae
vae_train = load_VAE_model(args.vae_checkpoint_path, args)
vae_train.set_expert(expert)
dtype = torch.FloatTensor

# forward expert trajectories
results = vae_train.test_generate_trajectory_variable_length(
                    expert,
                    num_test_samples=5,
                    test_goal_policy_only=False)

# save checkpoint
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
with open(args.results_pkl_path, 'wb') as results_f:
    pickle.dump(results, results_f, protocol=2)
    print('Did save results to {}'.format(args.results_pkl_path))

# plot with different color
# from matplotlib import pyplot as plt
# plt.savefig()