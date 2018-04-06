import numpy as np

import math
import torch
from torch.autograd import Variable

def epsilon_greedy_linear_decay(action_vector,
                                n_episodes,
                                n,
                                low=0.1,
                                high=0.9):
    if n <= n_episodes:
        eps = ((low-high)/n_episodes)*n + high
    else:
        eps = low

    if np.random.uniform() > eps:
        return np.argmax(action_vector)
    else:
        return np.random.randint(low=0, high=num_actions)

def epsilon_greedy(action_vector, eps=0.1):
    if np.random.uniform() > eps:
        return np.argmax(action_vector)
    else:
        return np.random.randint(low=0, high=num_actions)

def greedy(action_vector):
    return np.argmax(action_vector)

def oned_to_onehot(action_delta, n=num_actions):
    action_onehot = np.zeros(n,)
    action_onehot[int(action_delta)] = 1.0

    return action_onehot

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    pi_tensor = torch.FloatTensor(math.pi).type(x.type)
    log_density = -(x - mean).pow(2) / (2 * var) \
            - 0.5 * torch.log(2 * Variable(pi_tensor)) - log_std
    return log_density.sum(1)
