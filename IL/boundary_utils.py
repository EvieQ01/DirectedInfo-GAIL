

import numpy as np
import pdb



def get_all_count_dict(all_traj):
    '''
    input: list of (x, y), (N, L, 2) 
    output: dict{dict}
    -(x, y):
        - (x', y'): count
    '''
    all_count = {}
    for n in range(len(all_traj)):
        traj = all_traj[n]
        # pdb.set_trace()

        for time_step in range(traj.shape[0] - 1):
            xy_posi = (traj[time_step][0], traj[time_step][1])
            xy_posi_plus = (traj[time_step + 1][0], traj[time_step + 1][1])

            if xy_posi not in all_count.keys(): #(x, y)for the first time
                all_count[xy_posi] = {}
                all_count[xy_posi][xy_posi_plus] = 1
            elif xy_posi_plus not in all_count[xy_posi].keys(): # (x', y') for the first time
                all_count[xy_posi][xy_posi_plus] = 1
            else:# (x', y') add one count
                all_count[xy_posi][xy_posi_plus] += 1
    return all_count

def get_all_count_dict_reverse(all_traj):
    all_count = {}
    # reverse it 
    for n in range(len(all_traj)):
        traj = all_traj[n]
        for time_step in range(traj.shape[0] - 1):
            xy_posi = (traj[-time_step-1][0], traj[-time_step-1][1])
            xy_posi_plus = (traj[-time_step-2][0], traj[-time_step-2][1])
            if xy_posi not in all_count.keys(): #(x, y)for the first time
                all_count[xy_posi] = {}
                all_count[xy_posi][xy_posi_plus] = 0
            elif xy_posi_plus not in all_count[xy_posi].keys(): # (x', y') for the first time
                all_count[xy_posi][xy_posi_plus] = 0
            else:# (x', y') add one count
                all_count[xy_posi][xy_posi_plus] += 1
    return all_count

def get_boundary_from_all_traj(all_traj):
    boundary = []
    # pdb.set_trace()

    all_count_xy = get_all_count_dict(all_traj)
    for xy_key in all_count_xy:
        values = all_count_xy[xy_key]
        if len(values.keys()) != 1 : # not only one (x', y') to go, then means multiple choices.
            boundary.append(xy_key)
    all_count_xy = get_all_count_dict_reverse(all_traj)
    for xy_key in all_count_xy:
        values = all_count_xy[xy_key]
        if len(values.keys()) != 1 : # not only one (x', y') to go, then means multiple choices.
            boundary.append(xy_key)
    return boundary

