import pdb
from matplotlib import pyplot as plt
import numpy as np
import torch
from dataset_utils import dynamic_sub_traj_dataset, sub_traj
from models import QHead
# update every x epoch, use posteriorQ to update,
def update_dataset(sub_traj_dataset: dynamic_sub_traj_dataset, netQ:QHead, encoder, device, label_dtype, threshold=0.3)-> None:
    distances_of_c = []
    pre_context = None
    pre_traj_index = None
    # calculate distances
    with torch.no_grad():
        for i in range(sub_traj_dataset.cur_size): # for each subtraj
            subtraj = sub_traj_dataset.sample_batch_index(i)
            # pdb.set_trace()
            subtraj_feat = subtraj.data.to(device) # (B, len)
            subtraj_feat = encoder.s_embed(subtraj_feat.type(label_dtype))
            context = netQ(subtraj_feat)
            # save distance of c_j and c_{j + 1}
            if pre_context is not None:
                distances_of_c.append(torch.abs(context - pre_context).sum().item())# infinite norm
            # save previous timestep cotext and traj_indx
            if pre_traj_index == subtraj.traj_index or pre_traj_index == None:
                pre_context = context
            else:
                pre_context = None # now this sub_traj comes from a new traj.
            pre_traj_index = subtraj.traj_index
        print("=> len distances of [c_j] and [c_j+1] is ", len(distances_of_c))
        print("=> size of sub_traj_dataset is ", sub_traj_dataset.cur_size)
        print("=> size traj is ", sub_traj_dataset.origin_size)
        print("===> (1 = 2 - 3)")
        distances_of_c = np.array(distances_of_c)
        plt.hist(distances_of_c)
        plt.show()
    
    # renew subtraj for distance < 0.1[TODO]
    pre_subtraj = sub_traj_dataset.sample_batch_index(0)
    pre_subindex = 0
    pre_traj_index = 0
    all_subtraj_new = []
    for i in range(sub_traj_dataset.cur_size - 1): # for each subtraj
        # Note that skip every first one! So, + pre_subtraj.traj_index!
        subtraj = sub_traj_dataset.sample_batch_index(i + 1) # begin from j==1 (distances_of_c == c - c_{j-1})
        if pre_subtraj is None: # check next
            pre_subtraj = subtraj
            continue
        if pre_traj_index != subtraj.traj_index: # now this sub_traj comes from a new traj.
            pre_traj_index = subtraj.traj_index 
            pre_subindex = 0
            pre_subtraj = subtraj
            # pdb.set_trace()
            continue

        # judge by distance to concat
        if distances_of_c[i - pre_subtraj.traj_index] < threshold * distances_of_c.max(): # merge subtrajs that has small distance
            subtraj_new = sub_traj(data=torch.concat((pre_subtraj.data, subtraj.data)), sub_index=pre_subindex, traj_index=subtraj.traj_index)
            pre_subindex += 1 # marked as next subtraj
            all_subtraj_new.append(subtraj_new)
            pre_subtraj = None # already update, re-initialize pre_subtraj next time.
        else:
            all_subtraj_new.append(pre_subtraj)
            pre_subtraj = subtraj
        pre_traj_index = subtraj.traj_index 
    # pdb.set_trace()
    sub_traj_dataset.update(all_subtraj=all_subtraj_new)
    
