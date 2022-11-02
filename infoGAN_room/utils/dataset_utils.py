from copy import deepcopy
import pdb
import random
from re import L
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import int64



# fixed sub_traj_dataset
# created by boundary list
class sub_traj_dataset():
    def __init__(self, all_traj: tuple, boundary_list: list, max_len=10, batch_size=8, set_diff=None) -> None:
        '''
        all_traj: (N, L, Dim(s))
        boundary_list: [(x, y), ...]
        '''
        self.all_traj = all_traj
        self.boundary_list = boundary_list
        self.batch_size = batch_size

        self.all_sub_traj = []
        self.current_sample_index = 0
        def get_index_for_boundary(state):
            if (state[0], state[1]) in boundary_list:
                return boundary_list.index((state[0], state[1]))
            else:
                return -1
        for i in range(len(all_traj)):
            sub_traj_begin = 0
            sub_traj_end = 0
            current_s = (all_traj[i][sub_traj_end][0], all_traj[i][sub_traj_end][1])
            end_flag = False
            # compute for whole trajectory
            while end_flag == False:
                while (current_s not in boundary_list and sub_traj_end-sub_traj_begin < max_len) or sub_traj_begin == sub_traj_end:
                    # if didnt reach boundary and didnt reach max_len
                    sub_traj_end += 1
                    if sub_traj_end == len(all_traj[i]):
                        sub_traj_end -= 1
                        end_flag = True
                        break
                    current_s = (all_traj[i][sub_traj_end][0], all_traj[i][sub_traj_end][1])
                
                if sub_traj_end > sub_traj_begin:
                    #   add boundary_index:
                    begin_ind = get_index_for_boundary(all_traj[i][sub_traj_begin])
                    end_ind = get_index_for_boundary(all_traj[i][sub_traj_end])
                    self.all_sub_traj.append([torch.tensor(all_traj[i][sub_traj_begin : sub_traj_end]), [begin_ind, end_ind]])
                sub_traj_begin = sub_traj_end
        # count of all sub trajs
        self.size = len(self.all_sub_traj)
        print(f'=> end extracting all sub_trajs with size {self.size}')
        self.set_diff = set_diff.tolist()
        if set_diff is not None:
            # self.set_diff_onehot = [F.one_hot(torch.tensor(i), num_classes=len(set_diff)) for i in range(len(set_diff))]
            # # pdb.set_trace()
            # self.set_diff_onehot = torch.stack(self.set_diff_onehot)
            # self.set_diff_onehot = torch.eye(len(self.set_diff))
            self.turn_xy_posi_to_onehot()



    def turn_xy_posi_to_onehot(self, traj=None):
        if traj is None: # change all self.traj into one_hot
            all_onehot_sub_traj = []
            # sub_traj is List[[states], [begin_boundary_ind, end_boundary_ind]]
            # sub_traj[0] == [states]
            # sub_traj[1] == [begin_boundary_ind, end_boundary_ind]]
            for sub_traj in self.all_sub_traj:
                onehot_sub_traj = self._turn_xy_posi_to_onehot_single(sub_traj[0])
                all_onehot_sub_traj.append([onehot_sub_traj, sub_traj[1]])
            self.all_sub_traj = all_onehot_sub_traj
        else:# change specific traj into one_hot
            # Tensor (B, len, len(set_diff))
            return self._turn_xy_posi_to_onehot_single(traj)

    def _turn_xy_posi_to_onehot_single(self, sub_traj)-> Tensor:
        onehot_sub_traj = torch.zeros(len(sub_traj))#, len(self.set_diff)))
        for j in range(len(sub_traj)):
            xy_posi = sub_traj[j]
            # indices
            ind = self.set_diff.index(xy_posi.type(torch.int64).tolist())
            # turn into one_hot
            onehot_sub_traj[j] = torch.tensor(ind)#, num_classes=len(self.set_diff)
        return onehot_sub_traj

    def sample_padded_batch(self):
        if self.current_sample_index + self.batch_size >= self.size:
            random.shuffle(self.all_sub_traj)
            self.current_sample_index = 0
        sub_trajs = self.all_sub_traj[self.current_sample_index : self.current_sample_index + self.batch_size]
        # padd to same length
        sub_traj_padded = pad_sequence([s_ind_list[0] for s_ind_list in sub_trajs], batch_first=True, padding_value=len(self.set_diff)) # (B, max_L, dim(S))
        # # genearte mask
        # mask = torch.ones(self.batch_size, sub_traj_padded.shape[1])
        # for i in range(len(sub_trajs)):
        #     mask[i, sub_trajs[i].shape[0]:] = 0 # indices larger than true sequence length set to 0.

        boundary_index = torch.tensor([s_ind_list[1] for s_ind_list in sub_trajs]) # B, 2
        self.current_sample_index += self.batch_size
        # (B, max_L, dim(S)), (B, max_len)
        return sub_traj_padded, boundary_index#, mask

    def sample_packed_batch(self):
        if self.current_sample_index + self.batch_size >= self.size:
            random.shuffle(self.all_sub_traj)
            self.current_sample_index = 0
        sub_trajs = self.all_sub_traj[self.current_sample_index : self.current_sample_index + self.batch_size]
        # padd to same length
        # sub_traj_packed = pack_sequence(sub_trajs, enforce_sorted=False) # (B, max_L, dim(S))
        sub_traj_packed = pack_sequence([s_ind_list[0] for s_ind_list in sub_trajs], enforce_sorted=False) # (B, max_L, dim(S))
        boundary_index = torch.tensor([s_ind_list[1] for s_ind_list in sub_trajs]) # B, 2

        self.current_sample_index += self.batch_size
        # (B, max_L, dim(S)), (B, max_len)
        return sub_traj_packed, boundary_index

    def sample_specific_batch(self, index):
        sub_trajs = self.all_sub_traj[index : index + self.batch_size]
        # padd to same length
        # sub_traj_packed = pack_sequence(sub_trajs, enforce_sorted=False) # (B, max_L, dim(S))
        sub_traj_packed = pack_sequence([s_ind_list[0] for s_ind_list in sub_trajs], enforce_sorted=False) # (B, max_L, dim(S))
        boundary_index = torch.tensor([s_ind_list[1] for s_ind_list in sub_trajs]) # B, 2
        # (B, max_L, dim(S)), (B, max_len)
        return sub_traj_packed, boundary_index
    # def get_boundary_index_for_batch(self, packed_batch):


    def init_state_sample(self):
        assert self.set_diff is not None
        indices = random.sample(torch.arange(0, len(self.set_diff)).tolist(), k=self.batch_size)
        # (B, x_dim)
        return self.set_diff_onehot[indices]



class sub_traj():
    def __init__(self, data, traj_index, sub_index) -> None:
        '''
        [input]
            data: list
            traj_index: int
            sub_index: int (use for determine adjascency)
        '''
        self.data = data #[1, 5, 20, 4] index of (x, y) in set_diff
        self.traj_index = traj_index
        self.sub_index = sub_index

class dynamic_sub_traj_dataset():
    def __init__(self, all_traj: tuple, boundary_adjacent_list: list, all_adjacent_list:list,  max_len=10, batch_size=8, set_diff=None) -> None:
        '''
        all_traj: (N, L, Dim(s))
        '''
        self.all_traj = all_traj
        self.max_len = max_len
        self.batch_size = batch_size
        self.set_diff = set_diff.tolist()
        self.all_subtraj = []
        self.all_subtraj_ordered = []
        # pdb.set_trace()
        # xy_tuple is (end_state_of_ci, begin_state_of_ci+1)
        self.boundary_adjacent_list = [[self.xy_2_index(xy_tuple[0]), self.xy_2_index(xy_tuple[1])] for xy_tuple in  boundary_adjacent_list]
        self.all_adjacent_list = [[self.xy_2_index(xy_tuple[0]), self.xy_2_index(xy_tuple[1])] for xy_tuple in  all_adjacent_list]
        
        self.origin_size = len(all_traj)
        # initialize with seq_len == 1
        for traj_index in range(self.origin_size):
            for step in range(len(self.all_traj[traj_index])):
                xy_posi = self.all_traj[traj_index][step]
                # indices
                # pdb.set_trace()
                data = self.xy_2_index(xy_posi=xy_posi)
                self.all_subtraj.append(sub_traj(data=torch.tensor(data).reshape((1)), traj_index=traj_index, sub_index=step))
        self.cur_size = len(self.all_subtraj)
        self.step_size = len(self.all_subtraj)
        self.update(all_subtraj=self.all_subtraj)
        # print(f"=> Create dataset with {self.cur_size} subtraj ({self.cur_size} steps in all, for whole dataset).")
        # print('sample: ', self.all_subtraj[0])
        # self.all_subtraj_ordered = deepcopy(self.all_subtraj)

    # update all_subtraj and sizes
    def update(self, all_subtraj) -> None:
        self.cur_size = len(all_subtraj)
        self.all_subtraj = all_subtraj
        print(f"=> Update dataset with {self.cur_size} subtraj ({self.step_size} steps in all, for whole dataset).")
        print('sample: ', self.all_subtraj[0].data)
        self.all_subtraj_ordered = deepcopy(self.all_subtraj)
        self.current_sample_index = 0
    
    def xy_2_index(self, xy_posi):
        if isinstance(xy_posi, np.ndarray):
            return self.set_diff.index(xy_posi.astype(np.int64).tolist())
        else:# tuple or list
            return self.set_diff.index(np.array(xy_posi, dtype=np.int64).tolist())

    def sample_padded_batch_sorted(self):
        if self.current_sample_index + self.batch_size >= self.cur_size:
            random.shuffle(self.all_subtraj)
            self.current_sample_index = 0
        sub_trajs = self.all_subtraj[self.current_sample_index : self.current_sample_index + self.batch_size]
        # padd to same length
        batch = [sub_traj.data + 1 for sub_traj in sub_trajs] # !! +1 index
        length = torch.tensor([s.shape[0] for s in batch])
        sorted_seq_lengths, indices = torch.sort(length, descending=True)
        _  , desorted_indices = torch.sort(indices, descending=False)                

        # pdb.set_trace()
        # !! sort return value(boundary and padded batch)
        sub_traj_padded = pad_sequence(batch, batch_first=True, padding_value=0)[indices] # (B, max_L, dim(S))
        potential_boundary = (torch.stack((torch.tensor([sub_traj[0] for sub_traj in batch]) , \
                                            torch.tensor([sub_traj[-1] for sub_traj in batch]))) - 1).T[indices] # only use index = index + 1 when embedding, else use index in set_diff
        # sub_traj_padded = pad_sequence([s_ind_list[0] for s_ind_list in sub_trajs], batch_first=True, padding_value=len(self.set_diff)) # (B, max_L, dim(S))
        # boundary_index = torch.tensor([s_ind_list[1] for s_ind_list in sub_trajs]) # B, 2
        # self.current_sample_index += self.batch_size
        # (B, max_L, dim(S)), (B, 2, dim(S))
        return sub_traj_padded, potential_boundary, sorted_seq_lengths #, boundary_index#, mask

    def sample_batch_index(self, index):
        sub_traj = self.all_subtraj_ordered[index]
        # (B, dim(S))
        return sub_traj #, boundary_index#, mask
    
    def convert2_true_posi(self, index):
        sub_traj = self.all_subtraj_ordered[index]
        # (B, dim(S))
        # pdb.set_trace()
        return np.array([self.set_diff[set_diff_index] for set_diff_index in sub_traj.data]) # (L, 2)

    # @staticmethod
    def get_adjacent_mask(self, potential_boundary):
        '''
        input: (B, 2) Batches of starting and ending point states. As potential boundary.
        output: -1 when adjacent state is boundary,
                1  when xxx            is not boundary but adjacent.
        '''
        # get adjacency mask
        batch_size = potential_boundary.shape[0]
        view1 = torch.repeat_interleave(potential_boundary[:, 0], batch_size) # B^2 (11111..., 22222..., )
        view2 = potential_boundary[:, 1].repeat(batch_size) # B^2 (1:10, 1:10, ...)
        # mask = torch.bitwise_or(torch.bitwise_and((view1 == view2 ).reshape((batch_size, -1)), potential_boundary[:, 0] != -1), \
        #         torch.bitwise_and((view1 == view2 ).reshape((batch_size, -1)), potential_boundary[:, 1] != -1))
        mask = torch.stack((view2, view1), dim=-1) # (B, B, 2)
        mask = mask.reshape((-1, 2))# (B^2, 2)
        # each tuple in mask is (end_of_a_seq, begin_of_b_seq)
        i = 0
        adjacent_mask = torch.zeros(mask.shape[0])
        for e_b_tuple in mask:
            if e_b_tuple.tolist() in self.all_adjacent_list: #TODO
                adjacent_mask[i] = 1
            if e_b_tuple.tolist() in self.boundary_adjacent_list: #TODO
                adjacent_mask[i] = -1 # 
            i += 1
        adjacent_mask = adjacent_mask.reshape((batch_size, batch_size))
        # pdb.set_trace()
        return adjacent_mask + adjacent_mask.T # distance matrix should be symetric.

    def get_boundary_list(self, potential_boundary):
        '''
        
        '''
        pass

    def get_weight_for_divergence(self, potential_boundary):
        '''
        
        '''
        pass
# define fixed_noise_dataset.
class noise_dataset():
    def __init__(self, c_dim, z_dim, dtype) -> None:
        sample_count = 10 * c_dim
        z = torch.randn(sample_count, z_dim, 1)
        self.fixed_noise = z
        idx = np.arange(c_dim).repeat(10)
        # 100 * 1 * 10
        dis_c = torch.zeros(sample_count, c_dim)
        dis_c[torch.arange(0, sample_count), idx] = 1.0

        dis_c = dis_c.view(sample_count, -1, 1) # (100, 10, 1) 100 samples. uniform

        self.fixed_noise = torch.cat((self.fixed_noise, dis_c), dim=1).type(dtype=dtype)
    
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

