from copy import deepcopy
import pdb
import random
from re import L
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
import torch
import torch.nn.functional as F

class sub_traj():
    def __init__(self, data, traj_index=None, sub_index=None) -> None:
        '''
        [input]
            data: list
            traj_index: int
            sub_index: int (use for determine adjascency)
        '''
        self.data = data # (x, y) 
        self.traj_index = traj_index
        self.sub_index = sub_index
        self.div = None
    
    def set_divergence(self, div):
        # set divergence at last timestep.
        self.div = div

class dynamic_sub_traj_dataset():
    def __init__(self, all_traj: tuple,  max_len=10, batch_size=8, thresh=0.5, div=None) -> None:
        '''
        all_traj: (N, L, Dim(s))
        '''
        self.all_traj = [] # turn all_traj into 'sub_traj' type
        self.max_len = max_len
        self.batch_size = batch_size
        # self.set_diff = set_diff.tolist() # only used in discrete env
        self.all_subtraj = []
        self.all_subtraj_ordered = []
        # xy_tuple is (end_state_of_ci, begin_state_of_ci+1)
        
        # initialize divergence
        self.set_div_for_all(all_traj, div=div)
        self.origin_size = len(all_traj)
        self.current_sample_index = 0
        # initialize seq_len according to divergence
        subtraj = None
        for traj_index in range(self.origin_size):
            sub_index = 0
            for step in range(len(self.all_traj[traj_index])):
                xy_posi = self.all_traj[traj_index][step].data.unsqueeze(0)
                div_posi = self.all_traj[traj_index][step].div
                if subtraj is None:
                # initialize a new subtraj
                    subtraj = sub_traj(data=xy_posi, traj_index=traj_index, sub_index=sub_index)
                    continue

                if step == len(self.all_traj[traj_index]) - 1:
                # if xy is last timestep of a traj, append it by defalte
                    # if traj_index == 2:
                    #     pdb.set_trace()
                    self.all_subtraj.append(sub_traj(data=torch.cat((subtraj.data, xy_posi)), traj_index=traj_index, sub_index=sub_index))
                    print('=> Append ', torch.cat((subtraj.data, xy_posi)).shape, traj_index)
                    subtraj = None
                    continue
                
                if div_posi < thresh:
                # concat
                    subtraj = sub_traj(data=torch.cat((subtraj.data, xy_posi)), traj_index=traj_index, sub_index=sub_index)
                else:                   
                # append and begin from new
                    subtraj = sub_traj(data=torch.cat((subtraj.data, xy_posi)), traj_index=traj_index, sub_index=sub_index)
                    self.all_subtraj.append(subtraj)
                    print('=> Append ', subtraj.data.shape, traj_index)
                    sub_index += 1
                    subtraj = None
        self.all_subtraj_ordered = deepcopy(self.all_subtraj)
        # pdb.set_trace()
        self.cur_size = len(self.all_subtraj)
    
    def set_div_for_all(self, all_traj, div) -> None:
        index = 0
        # pdb.set_trace()
        for i in range(len(all_traj)):
            self.all_traj.append([])
            for j in range(len(all_traj[i])):
                self.all_traj[i].append(sub_traj(data=torch.tensor(all_traj[i][j])))
                self.all_traj[i][-1].set_divergence(div[index])
                index += 1
        print(index)
    
    def sample_padded_batch_sorted(self,):
        if self.current_sample_index + self.batch_size >= self.cur_size:
            random.shuffle(self.all_subtraj)
            self.current_sample_index = 0
        sub_trajs = self.all_subtraj[self.current_sample_index : self.current_sample_index + self.batch_size]
        # padd to same length
        batch = [sub_traj.data for sub_traj in sub_trajs] # !! +1 index
        length = torch.tensor([s.shape[0] for s in batch])
        sorted_seq_lengths, indices = torch.sort(length, descending=True)
        _  , desorted_indices = torch.sort(indices, descending=False)                

        # pdb.set_trace()
        # !! sort return value(boundary and padded batch)
        sub_traj_padded = pad_sequence(batch, batch_first=True, padding_value=0)[indices] # (B, max_L, dim(S))
        potential_boundary = torch.stack((torch.stack([sub_traj[0] for sub_traj in batch])[indices], torch.stack([sub_traj[-1] for sub_traj in batch])[indices] ))# (2, B, dim(s))
        # sub_traj_padded = pad_sequence([s_ind_list[0] for s_ind_list in sub_trajs], batch_first=True, padding_value=len(self.set_diff)) # (B, max_L, dim(S))
        # boundary_index = torch.tensor([s_ind_list[1] for s_ind_list in sub_trajs]) # B, 2
        self.current_sample_index += self.batch_size
        # (B, max_L, dim(S)), (B, 2, dim(S))
        return sub_traj_padded, potential_boundary, sorted_seq_lengths #, boundary_index#, mask
    
    # update all_subtraj and sizes
    def update(self, all_subtraj) -> None:
        self.cur_size = len(all_subtraj)
        self.all_subtraj = all_subtraj
        print(f"=> Update dataset with {self.cur_size} subtraj ({self.step_size} steps in all, for whole dataset).")
        print('sample: ', self.all_subtraj[0].data)
        self.all_subtraj_ordered = deepcopy(self.all_subtraj)
        self.current_sample_index = 0
    
    def sample_batch_index(self, index):
        sub_traj = self.all_subtraj_ordered[index]
        # (L, dim(S))
        return sub_traj #, boundary_index#, mask

        
    
def noise_sample(dis_c_dim, z_dim, batch_size, dtype):
    """
    Sample random noise vector for training.

    INPUT
    --------
    dis_c_dim : Dimension of discrete latent code.
    z_dim : Dimension of iicompressible noise.
    batch_size : Batch Size
    dtype : GPU/CPU
    
    
    Output:

    """

    c_idx = np.random.randint(dis_c_dim, size=batch_size)
    z_idx =  torch.randn(batch_size, z_dim)
    # return noise, idx
    # (B, dim), (B)
    return z_idx, torch.tensor(c_idx)

