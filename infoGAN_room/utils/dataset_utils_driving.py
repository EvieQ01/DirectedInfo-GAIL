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
    def __init__(self, data, traj_index, sub_index) -> None:
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
    def __init__(self, all_traj: tuple,  max_len=10, batch_size=8, thresh=0.1) -> None:
        '''
        all_traj: (N, L, Dim(s))
        '''
        self.all_traj = all_traj
        self.max_len = max_len
        self.batch_size = batch_size
        # self.set_diff = set_diff.tolist() # only used in discrete env
        self.all_subtraj = []
        self.all_subtraj_ordered = []
        # pdb.set_trace()
        # xy_tuple is (end_state_of_ci, begin_state_of_ci+1)
        self.origin_size = len(all_traj)
        # initialize seq_len according to divergence
        subtraj = None
        for traj_index in range(self.origin_size):
            sub_index = 0
            for step in range(len(self.all_traj[traj_index])):
                xy_posi = self.all_traj[traj_index][step]
                if subtraj is None:
                # initialize a new subtraj
                    subtraj = sub_traj(data=self.all_traj[traj_index][step], traj_index=traj_index, sub_index=sub_index)
                    continue

                if step != len(self.all_traj[traj_index]) - 1:
                # if xy is last timestep of a traj, append it by defalte
                    self.all_subtraj.append(sub_traj(data=torch.cat((subtraj.data, xy_posi)), traj_index=traj_index, sub_index=sub_index))
                    continue
                
                if xy_posi.div < thresh:
                # concat
                    subtraj = sub_traj(data=torch.cat((subtraj.data, xy_posi)), traj_index=traj_index, sub_index=sub_index)
                else:                   
                # append and begin from new
                    subtraj = sub_traj(data=torch.cat((subtraj.data, xy_posi)), traj_index=traj_index, sub_index=sub_index)
                    self.all_subtraj.append(subtraj)
                    sub_index += 1
                    subtraj = None

        self.cur_size = len(self.all_subtraj)

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
        # (B, dim(S))
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

    # z = torch.randn(batch_size, z_dim)

    # idx = np.zeros((batch_size))
    # dis_c = torch.zeros(batch_size, dis_c_dim)
    
    # idx = np.random.randint(dis_c_dim, size=batch_size)
    # dis_c[torch.arange(0, batch_size),  idx] = 1.0

    # dis_c = dis_c.view(batch_size, -1)

    # noise = z
    # noise = torch.cat((z, dis_c), dim=1).type(dtype=dtype)

    c_idx = np.random.randint(dis_c_dim, size=batch_size)
    z_idx = np.random.randint(z_dim, size=batch_size)
    # return noise, idx
    return torch.tensor(z_idx), torch.tensor(c_idx)

