import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import pdb
#import ipdb
import h5py
import importlib
import pickle
import math

sns.set()
from load_expert_traj import recursively_save_dict_contents_to_group

# %pylab inline
# inline doesn't give interactive plots
# %matplotlib inline 
# %matplotlib notebook
plt.rcParams['figure.figsize'] = (6.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'Blues'
from utils.entropy_estimator import get_h, KDTree
def get_mapping(all_traj):
    '''
    output: (map, all_states)
        map : dict {index -> (n, timestep)}
        all_states : (all_numbers, 2)
    '''
    mapping_dict = {}
    index = 0
    all_states = []
    for i in range(len(all_traj)):
        episode_len = all_traj[i].shape[0]
        for j in range(episode_len):
            all_states.append(all_traj[i][j])
            mapping_dict[index] = (i, j)
            index += 1
    return mapping_dict, np.array(all_states)
    # return mapping_dict, np.concatenate(all_states).reshape((-1, 2))

def get_neighbor_states_of_s(all_states, k=5):
    '''
    input: (all_numbers, 2)
    output: list[(nearest_k_indices)] with len=all_numbers
    '''
    kdtree = KDTree(all_states)
    all_indices = []
    all_distances = []
    # for i in range(all_states.shape[0]):
    distances, indices = kdtree.query(all_states, k + 1, distance_upper_bound=0.15)
    # all_indices.append(indices)
    # all_distances.append(distances)
    # return all_indices, all_distances
    return indices[:, 1:], distances[:, 1:] # ommited point x itself


def get_H_for_all_states(all_traj, delta_t=5):
    '''
    input: list of (x, y), (N, L, 2) 
    output: (all_states_count): float, Entropy
    '''
    all_states_H = []
    all_states_H_inverse = []
    mapping_dict, all_states = get_mapping(all_traj=all_traj)
    all_indices, all_distances = get_neighbor_states_of_s(all_states=all_states)
    # pdb.set_trace()
    for indices in all_indices:
        # Get neighbor of k-nearst for each state s.
        neighbors = []
        neighbors_inverse = []
        for idx in indices:
            if idx == all_states.shape[0]:
                # ommited neibor
                continue
            i, j = mapping_dict[idx]
            neighbors.append(all_traj[i][j: j+delta_t])
            neighbors_inverse.append(all_traj[i][max(j - delta_t, 0): j])
        # Get entropy of (k * delta_t, 2) states
        entropy = get_h(np.concatenate(neighbors), k=1, min_dist=1e-10) 
        entropy_inverse = get_h(np.concatenate(neighbors), k=1, min_dist=1e-10) 
        all_states_H.append(entropy)
        all_states_H_inverse.append(entropy_inverse)

    # pdb.set_trace()
    all_entropy = np.max(np.stack((np.array(all_states_H), np.array(all_states_H_inverse)), axis=0), axis=0)
    all_entropy = np.array(all_entropy)
    all_entropy -= all_entropy.min()
    all_entropy /= all_entropy.max() # noramalized into [0, 1]
    plot_entropy(all_states=all_states, all_entropy=all_entropy)
    hist = np.histogram(a=all_entropy, bins=np.arange(0, 1, 0.1))
    print(hist)
    return all_entropy

def plot_entropy(all_states, all_entropy, plot_title='entropy'):
    use_color = 'brown'
    colors = ["lightgreen", use_color]
    cmap = LinearSegmentedColormap.from_list("cmap", colors)
    # clist = [(0, 'white'), (1, use_color)]
    # colors_ = LinearSegmentedColormap.from_list("", clist)
    # cmappable = ScalarMappable(norm=Normalize(0,1), cmap=colors_)    
    # colors = cm.rainbow(np.linspace(0, 1, len(x_list)))
    # for state, h in zip(all_states, all_entropy):
    plt.scatter(all_states[:, 0], all_states[:, 1], linewidth=5, alpha=0.1, c=cmap(all_entropy))
    plt.axis('equal')    
    plt.title(f"States with {plot_title}")
    plt.show()

def get_max_dist_for_all_states(all_traj, delta_t, neighbor_k):
    '''
    input: list of (x, y), (N, L, 2) 
    output: (all_states_count): float, Entropy
    '''
    all_states_H = []
    all_states_H_inverse = []
    mapping_dict, all_states = get_mapping(all_traj=all_traj)
    all_indices, all_distances = get_neighbor_states_of_s(all_states=all_states, k=neighbor_k)
    # pdb.set_trace()
    # repeat the last state to get same length
    all_traj = list(all_traj)
    for i in range(len(all_traj)):
        padded_states = np.repeat(all_traj[i][-1], delta_t - 1).reshape((-1, 2), order='F')
        all_traj[i] = np.concatenate((all_traj[i], padded_states.copy()), axis=0)
    for indices in all_indices:
        # Get neighbor of k-nearst for each state s.
        neighbors = []
        neighbors_inverse = []
        for idx in indices:
            if idx == all_states.shape[0]:
                # ommited neibor too far
                continue
            i, j = mapping_dict[idx]
            neighbors.append(all_traj[i][j: j+delta_t])
            if j < delta_t:
                continue
            neighbors_inverse.append(all_traj[i][j - delta_t: j])
        # Get entropy of (k * delta_t, 2) states
        # entropy = get_h(np.concatenate(neighbors), k=1, min_dist=1e-10) 
        # entropy_inverse = get_h(np.concatenate(neighbors), k=1, min_dist=1e-10) 
        # Compute distance matrix for each state with k neighbors each with Len * 2dim
        # pdb.set_trace()
        neighbors = np.stack(neighbors, axis=0)
        look_1 = np.expand_dims(neighbors, axis=0) # 1 * k * 1000 * 2
        look_2 = np.expand_dims(neighbors, axis=1) # k * 1 * 1000 * 2
        # dist_matrix = np.sum(abs(look_1 - look_2), axis=-1)  # k * k * 1000
        # dist_matrix = np.mean(dist_matrix, axis=-1)
        dist_matrix = np.linalg.norm(look_1 - look_2, axis=-1)# k* k
        all_states_H.append(np.max(dist_matrix[0]))

        if len(neighbors_inverse) == 0:
            all_states_H_inverse.append(0.)
        else:
            neighbors = np.stack(neighbors_inverse, axis=0)
            look_1 = np.expand_dims(neighbors, axis=0) # 1 * k * 1000 * 2
            look_2 = np.expand_dims(neighbors, axis=1) # k * 1 * 1000 * 2
            # dist_matrix = np.sum(abs(look_1 - look_2), axis=-1)  # k * k * 1000
            # dist_matrix = np.mean(dist_matrix, axis=-1)
            dist_matrix = np.linalg.norm(look_1 - look_2, axis=-1)# k* k
            all_states_H_inverse.append(np.max(dist_matrix[0]))
    # pdb.set_trace()
    # all_entropy = np.array(all_states_H)
    all_entropy = np.max(np.stack((np.array(all_states_H), np.array(all_states_H_inverse)), axis=0), axis=0)
    all_entropy[all_entropy > 1] = 1.
    all_entropy[all_entropy < 0.5] = 0.
    # all_entropy -= all_entropy.min()
    # all_entropy /= all_entropy.max() # noramalized into [0, 1]
    plot_entropy(all_states=all_states, all_entropy=all_entropy, plot_title='divergence')
    hist = np.histogram(a=all_entropy, bins=np.arange(0, 1, 0.1))
    print(hist)
    return all_entropy

def get_boundary_from_all_traj_continuous(all_traj, delta_t=10, neighbor_k=10):
    '''
    input: list of (x, y), (N, L, 2) 
    output: list of (x, y), shape = (topk, 2)
    '''
    boundary = []
    all_entropy = get_H_for_all_states(all_traj=all_traj, delta_t=delta_t)
    # all_max_dist = get_max_dist_for_all_states(all_traj=all_traj, delta_t=delta_t, neighbor_k=neighbor_k)
    # hist = np.histogram(a=all_entropy, bins=np.arange(0, 1, 0.1))
    # print(hist)
    # plot_entropy()
    # pdb.set_trace()

    return boundary

