import os
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_trajectory_room(eposode, traj_context_data, grid_size,
                    color_map=['black', 'grey', 'red'],
                    figsize=(6,6),
                    obstacles=None,
                    bounds=[-100, -.5, .5, 100],
                    save_path='', wandb_log_stamp=False):
    cmap = matplotlib.colors.ListedColormap(color_map)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Get context colors
    # context_colors = ['g', 'blue', 'y', 'black', 'k', 'w','c', 'm', 'red','orange']
    context_colors = ['black', 'gray', 'lightgrey', 'lightcoral', 'firebrick',\
                        'red', 'saddlebrown', 'orange', 'gold', 'yellow', \
                        'yellowgreen', 'lawngreen','green','cyan', 'deepskyblue', \
                        'royalblue', 'mediumslateblue', 'blueviolet', 'violet', 'fuchsia']
    i = 0
    states = traj_context_data['true_traj_state'] # num, len, 2
    context = traj_context_data['pred_context']# num, len, 10
    # pdb.set_trace()
    for i in range(len(states)):
        traj = states[i] # (num_sample, len, dim=2)
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
        pred_contexts = []
        for j in range(traj.shape[0]):
            x, y = traj[j, 0], traj[j, 1]
            marker = '.' if j > 0 else '*'
            assert np.argmax(context[i][j]) < len(context_colors)
            pred_color = context_colors[np.argmax(context[i][j])]
            pred_contexts.append(np.argmax(context[i][j]))
                
            ax.scatter(x=x + 0.5, y=y + 0.5, c=pred_color,
                    s=120, marker=marker, edgecolors='none')
        print("true:",traj)
        print('pred context', pred_contexts)

        if len(save_path) > 0:
            fig.tight_layout()
            if wandb_log_stamp:
                # single_fig_path = os.path.join(save_path, f'{wandb_log_stamp}_test_context_for_traj{i}.png')
                # fig.savefig(single_fig_path)
                # img = wandb.Image(single_fig_path)
                # wandb.log({f"episode{eposode}": img})
                wandb.log({f"episode{eposode}":wandb.Image(fig)})

            else:
                single_fig_path = os.path.join(save_path, f'{wandb_log_stamp}_test_context_for_traj{i}.png')
                fig.savefig(single_fig_path)
            # os.makedirs(single_fig_path)
            plt.close(fig)
    

def plot_trajectory_circle(eposode, traj_context_data, figsize=(6,6),
                    save_path='', wandb_log_stamp=False):

    # Get context colors
    # context_colors = ['g', 'blue', 'y', 'black', 'k', 'w','c', 'm', 'red','orange']
    context_colors = ['black', 'gray', 'lightgrey', 'lightcoral', 'firebrick',\
                        'red', 'saddlebrown', 'orange', 'gold', 'yellow', \
                        'yellowgreen', 'lawngreen','green','cyan', 'deepskyblue', \
                        'royalblue', 'mediumslateblue', 'blueviolet', 'violet', 'fuchsia']
    
    i = 0
    states = traj_context_data['true_traj_state'] # num, len, 2
    context = traj_context_data['pred_context']# num, len, 10
    # pdb.set_trace()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)
    ax.set_xlim(left=-6, right=6)
    ax.set_ylim(bottom=-6, top=6.)
    for i in range(len(states)):
        traj = states[i] # (num_sample, len, dim=2)
        pred_contexts = []
        for j in range(traj.shape[0]):
            x, y = traj[j, 0], traj[j, 1]
            marker = '.' if j > 0 else '*'
            assert np.argmax(context[i][j]) < len(context_colors)
            pred_color = context_colors[np.argmax(context[i][j])]
            pred_contexts.append(np.argmax(context[i][j]))
                
            ax.scatter(x=x , y=y , c=pred_color,
                    s=120, marker=marker, edgecolors='none')
        # print("true:",traj)
        print('pred context', pred_contexts)

    if len(save_path) > 0:
        fig.tight_layout()
        if wandb_log_stamp:
            # single_fig_path = os.path.join(save_path, f'{wandb_log_stamp}_test_context_for_traj{i}.png')
            # fig.savefig(single_fig_path)
            # img = wandb.Image(single_fig_path)
            # wandb.log({f"episode{eposode}": img})
            wandb.log({f"episode{eposode}":wandb.Image(fig)})

        else:
            single_fig_path = os.path.join(save_path, f'{wandb_log_stamp}_test_context_for_traj.png')
            fig.savefig(single_fig_path)
        # os.makedirs(single_fig_path)
        plt.close(fig)
