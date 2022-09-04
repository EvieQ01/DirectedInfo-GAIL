

import numpy as np
import pandas as pd
import json
import sys
import os
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pdb
#import ipdb
import h5py
import importlib
import pickle
import math
# results_file = args.results_pkl_path
discrete_context = True
plot_type = 'pred_context'

def get_multiple_traj_idx_from_goal_list(goal_list):
    last_goal_idx = 0
    goal_idx_list = []
    while last_goal_idx < len(goal_list):
        curr_goal_idx = last_goal_idx
        while curr_goal_idx < len(goal_list):
            if goal_list[last_goal_idx] == goal_list[curr_goal_idx]:
                curr_goal_idx = curr_goal_idx + 1
            else:
                break
        # we have one trajectory
        goal_idx_list.append((last_goal_idx, curr_goal_idx))
        last_goal_idx = curr_goal_idx
        
    return goal_idx_list

def plot_trajectory(traj_data, grid_size,
                    pred_traj_data=None,
                    obstacles=None,
                    rooms=None,
                    pred_context=[],
                    pred_context_discrete=False,
                    bounds=[-100, -.5, .5, 100],
                    color_map=['black', 'grey', 'red'],
                    save_path='', figsize=(6,6), show_expert=False):
    cmap = matplotlib.colors.ListedColormap(color_map)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    img_arr = np.ones(grid_size)
    if obstacles is not None:
        for o in obstacles:
            img_arr[o[1], o[0]] = 0.0
    ax.imshow(img_arr, 
              cmap=cmap,
              norm=norm,
              extent=[0, grid_size[1], 0, grid_size[0]],
              interpolation='none', aspect='equal', origin='lower',
              )

    ax.set_xticks(range(grid_size[1]))
    ax.set_yticks(range(grid_size[0]))
    ax.grid(True)
    
    # Get context colors
    if pred_context_discrete and len(pred_context) > 0:
        num_discrete = len(pred_context[0])
        context_colors = ['g', 'blue', 'y', 'black', 'k', 'w','c', 'm', 'red','orange']
    else:
        context_colors = ['g']

    for i in range(traj_data.shape[0]):
        x, y = traj_data[i, 0], traj_data[i, 1]
        print("true:",x, y)
        marker = '.' if i > 0 else '*'
        if show_expert:
            ax.scatter(x=x + 0.5, y=y + 0.5, c='r',
                        s=120, marker=marker, edgecolors='none')
        
        if pred_traj_data is not None:
            pred_x, pred_y = pred_traj_data[i, 0], pred_traj_data[i, 1]
            pred_color = context_colors[0]
            if len(context_colors) > 1:
                print('pred context', np.argmax(pred_context[i]))
                assert np.argmax(pred_context[i]) < len(context_colors)
                pred_color = context_colors[np.argmax(pred_context[i])]
                
            ax.scatter(x=pred_x + 0.5, y=pred_y + 0.75, c=pred_color,
                       s=120, marker=marker, edgecolors='none')

    if len(save_path) > 0:
        fig.savefig(save_path)
    
    fig.tight_layout()
    fig.savefig(save_path)
    
def softmax(x):
    if len(x.shape) == 2:
        new_x = x - np.max(x,axis=1)[:, np.newaxis]
        denom = np.sum(np.exp(new_x), axis = 1)[:, np.newaxis]
        return np.exp(new_x)/denom
    elif len(x.shape) == 1:
        new_x = x - np.max(x)
        denom = np.sum(np.exp(new_x))
        return np.exp(new_x) / denom
    else:
        raise ValueError("incorrect softmax input")

def plot_pickle_results(results_pkl_path, obstacles, rooms,
                        num_traj_to_plot=1):
    assert os.path.exists(results_pkl_path), \
        'results pickle does not exist {}'.format(results_pkl_path)
    with open(results_pkl_path, 'rb') as results_f:
        results_dict = pickle.load(results_f)
        
        total_traj = len(results_dict['true_traj_state'])
        num_plot_traj = 0
        # pdb.set_trace()
        while num_plot_traj < num_traj_to_plot:
            # traj_idx = np.random.randint(total_traj)
            traj_idx = num_plot_traj #+ 10

            goal_idx_list = get_multiple_traj_idx_from_goal_list(
                np.squeeze(results_dict['true_goal'][traj_idx]).tolist())
            # goal_idx_list = [(0, 50)]

            for goal_start_idx, goal_end_idx in goal_idx_list:
            
                traj_len = goal_end_idx - goal_start_idx
                true_traj, pred_traj = [], []
                pred_context_list = []
                          
                for j in range(traj_len):
                    x_true = (results_dict['true_traj_state'][traj_idx][goal_start_idx + j, 0, :]).tolist()
                    #TODO
                    x_pred = (results_dict['true_traj_state'][traj_idx][goal_start_idx + j, 0, :]).tolist()
                    true_traj.append(x_true)
                    pred_traj.append(x_pred)
                    if results_dict.get(plot_type) is not None:
                        if discrete_context:
                            pred_context = softmax(
                                results_dict[plot_type][traj_idx][goal_start_idx+j, 0, :]
                            )
                        else:
                            pred_context = results_dict[plot_type][traj_idx][goal_start_idx+j, 0, :]
                        pred_context_list.append(pred_context)
                
                # Plot trajectory
                fig_dir = os.path.join(os.path.dirname(results_pkl_path), 'visualize')
                os.makedirs(fig_dir,exist_ok='True')
                plot_trajectory(np.array(true_traj),
                                (15, 11),
                                pred_traj_data=np.array(pred_traj),
                                color_map=sns.color_palette("Blues_r"),
                                figsize=(6, 6),
                                obstacles=obstacles,
                                rooms=rooms,
                                pred_context=pred_context_list,
                                pred_context_discrete=discrete_context,
                                save_path=os.path.join(fig_dir, f'traj{num_plot_traj}.png')
                               )
                num_plot_traj += 1

def plot_pickle_results_context(results_pkl_path, obstacles, rooms,
                        num_traj_to_plot=1):
    assert os.path.exists(results_pkl_path), \
        'results pickle does not exist {}'.format(results_pkl_path)
    with open(results_pkl_path, 'rb') as results_f:
        results_dict = pickle.load(results_f)
        
        num_plot_traj = 0
        # pdb.set_trace()
        while num_plot_traj < num_traj_to_plot:
            # traj_idx = np.random.randint(total_traj)
            traj_idx = num_plot_traj #+ 10

            goal_idx_list = [(0, results_dict['true_traj_state'][traj_idx].shape[0])]
            # goal_idx_list = [(0, 50)]

            for goal_start_idx, goal_end_idx in goal_idx_list:
            
                traj_len = goal_end_idx - goal_start_idx
                true_traj, pred_traj = [], []
                pred_context_list = []
                          
                for j in range(traj_len):
                    x_true = (results_dict['true_traj_state'][traj_idx][goal_start_idx + j, 0, :]).tolist()
                    #TODO
                    x_pred = (results_dict['true_traj_state'][traj_idx][goal_start_idx + j, 0, :]).tolist()
                    true_traj.append(x_true)
                    pred_traj.append(x_pred)
                    if results_dict.get(plot_type) is not None:
                        if discrete_context:
                            pred_context = softmax(results_dict[plot_type][traj_idx][goal_start_idx+j, 0, :])
                        else:
                            pred_context = results_dict[plot_type][traj_idx][goal_start_idx+j, 0, :]
                        pred_context_list.append(pred_context)
                
                # Plot trajectory
                fig_dir = os.path.join(os.path.dirname(results_pkl_path), 'visualize')
                os.makedirs(fig_dir,exist_ok='True')
                plot_trajectory(np.array(true_traj),
                                (15, 11),
                                pred_traj_data=np.array(pred_traj),
                                color_map=sns.color_palette("Blues_r"),
                                figsize=(6, 6),
                                obstacles=obstacles,
                                rooms=rooms,
                                pred_context=pred_context_list,
                                pred_context_discrete=discrete_context,
                                save_path=os.path.join(fig_dir, f'traj{num_plot_traj}.png')
                               )
                num_plot_traj += 1

# plot_pickle_results(results_file, obstacles, rooms, num_traj_to_plot=1)