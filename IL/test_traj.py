
from load_expert_traj import Expert, ExpertHDF5, CircleExpertHDF5
from boundary_utils import get_boundary_from_all_traj
expert_path = './h5_trajs/room_trajs/traj_room_centre_len_50'
expert_path = './h5_trajs/room_trajs/traj_len_16'
vae_state_size = 4
expert = ExpertHDF5(expert_path, vae_state_size)
expert.push(only_coordinates_in_state=True, one_hot_action=True)
traj_expert = expert.sample_all()
state_expert, action_expert, c_expert, _ = traj_expert
boundary = get_boundary_from_all_traj(state_expert)