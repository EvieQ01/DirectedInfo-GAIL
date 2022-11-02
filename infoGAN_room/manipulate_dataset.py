from copy import deepcopy
import pdb
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils.dataset_utils import dynamic_sub_traj_dataset, sub_traj
from models import QHead
# update every x epoch, use posteriorQ to update,
def update_dataset(sub_traj_dataset: dynamic_sub_traj_dataset, netQ:QHead, encoder, device, label_dtype, threshold=0.3,max_len=5)-> None:
    distances_of_c = []
    pre_context = None
    pre_traj_index = None
    netQ.eval()
    distance_far = []
    distance_near = []
    pre_data = None
    # calculate distances
    with torch.no_grad():
        for i in range(sub_traj_dataset.cur_size): # for each subtraj
            subtraj = sub_traj_dataset.sample_batch_index(i)
            # pdb.set_trace()
            subtraj_feat = subtraj.data.to(device).unsqueeze(0) # (B, len)
            subtraj_feat = encoder.s_embed(subtraj_feat.type(label_dtype))
            context = torch.softmax(netQ(subtraj_feat), dim=-1).squeeze(0)
            if pre_context is not None:
                if pre_traj_index == subtraj.traj_index:
                    # save distance of c_j and c_{j + 1}
                    id1 = pre_context.argmax()
                    id2 = context.argmax()
                    # pdb.set_trace()
                    distances_of_c.append((torch.abs(context[id1] - pre_context[id1]) + \
                        torch.abs(context[id2] - pre_context[id2])).item())# L1 norm
                    e_b_tuple = [pre_data[-1].item(), subtraj.data[0].item()]
                    # if distances_of_c[-1] > 0.0005:
                    #     pdb.set_trace()
                    if e_b_tuple in sub_traj_dataset.boundary_adjacent_list:
                        distance_far.append(distances_of_c[-1])
                    else:
                        distance_near.append(distances_of_c[-1])

                # otherwise, this sub_traj comes from a new traj.
            
            # save previous timestep cotext and traj_indx
            pre_context = context
            pre_traj_index = subtraj.traj_index
            pre_data = subtraj.data

        print("=> len distances of [c_j] and [c_j+1] is ", len(distances_of_c))
        print("=> size of sub_traj_dataset is ", sub_traj_dataset.cur_size)
        print("=> size traj is ", sub_traj_dataset.origin_size)
        print("===> (1 = 2 - 3)")
        fig = plt.figure()
        distance_near = np.array(distance_near)
        distance_far = np.array(distance_far)
        plt.hist(distance_near)
        plt.hist(distance_far)
        print("fig:",fig)
        # plt.savefig('hist_of_distance_c_nearfar.png')
        # plt.show()
        # pdb.set_trace()
    
    # renew subtraj for distance < threshold * distances_of_c.max()
    pre_subtraj = sub_traj_dataset.sample_batch_index(0)
    pre_subindex = 0
    pre_traj_index = 0
    all_subtraj_new = []
    subtraj_new = deepcopy(pre_subtraj)
    for i in range(sub_traj_dataset.cur_size - 1): # for each subtraj
        # Note that skip every first one! So, + pre_subtraj.traj_index!
        subtraj = sub_traj_dataset.sample_batch_index(i + 1) # begin from j==1 (distances_of_c == c - c_{j-1})
        if pre_traj_index != subtraj.traj_index: # now this sub_traj comes from a new traj.
            # if the last time step is not concated, add it to new dataset
            if subtraj_new is not None:
                all_subtraj_new.append(deepcopy(subtraj_new))
                print('append1', pre_traj_index, [sub_traj_dataset.set_diff[data.item()] for data in subtraj_new.data])
                subtraj_new = deepcopy(subtraj)
            # elif pre_subtraj is not None:
            #     all_subtraj_new.append(pre_subtraj)
            #     print('append', pre_traj_index, [sub_traj_dataset.set_diff[data.item()] for data in pre_subtraj.data])
            pre_traj_index = subtraj.traj_index 
            pre_subindex = 0
            pre_subtraj = subtraj
            # pdb.set_trace()
            continue

        # judge by distance to concat
        # if distances_of_c[i - pre_subtraj.traj_index] < threshold * distances_of_c.max(): # merge subtrajs that has small distance
        #     subtraj_new = sub_traj(data=torch.cat((pre_subtraj.data, subtraj.data)), sub_index=pre_subindex, traj_index=subtraj.traj_index)
        #     pre_subindex += 1 # marked as next subtraj
        #     all_subtraj_new.append(subtraj_new)
        #     pre_subtraj = None # already update, re-initialize pre_subtraj next time.
        # else:
        #     all_subtraj_new.append(pre_subtraj)
        #     pre_subtraj = subtraj
        # pre_traj_index = subtraj.traj_index 
        _len = subtraj_new.data.shape[0] +  subtraj.data.shape[0]

        if distances_of_c[i - pre_subtraj.traj_index] < np.percentile(distances_of_c, int(100 * threshold)) and _len < max_len : # merge subtrajs that has small distance
            subtraj_new =  sub_traj(data=torch.cat((subtraj_new.data, subtraj.data)), sub_index=pre_subindex, traj_index=subtraj.traj_index)
        else: #large distance:
            all_subtraj_new.append(deepcopy(subtraj_new))
            print('append2', pre_traj_index, [sub_traj_dataset.set_diff[data.item()] for data in subtraj_new.data])
            pre_subindex += 1 # marked as next subtraj
            subtraj_new = deepcopy(subtraj) # empty new sub_traj
        pre_subtraj = subtraj #re-initialize pre_subtraj next time.
        # else: # single point
        #     all_subtraj_new.append(pre_subtraj)
        #     print('append', pre_traj_index, [sub_traj_dataset.set_diff[data.item()] for data in pre_subtraj.data])
        #     pre_subindex += 1 # marked as next subtraj
        #     pre_subtraj = subtraj # shift one timestep: append previous subtraj.
        pre_traj_index = subtraj.traj_index 
    

    if subtraj_new is not None:
        all_subtraj_new.append(deepcopy(subtraj_new))
        print('append3', pre_traj_index, [sub_traj_dataset.set_diff[data.item()] for data in subtraj_new.data])
    elif pre_subtraj is not None:
        all_subtraj_new.append(deepcopy(pre_subtraj))
        print('append4', pre_traj_index, [sub_traj_dataset.set_diff[data.item()] for data in pre_subtraj.data])

    # pdb.set_trace()
    netQ.train()
    sub_traj_dataset.update(all_subtraj=all_subtraj_new)
    return np.array(distances_of_c).min(), fig
