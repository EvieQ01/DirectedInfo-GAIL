import copy
import pdb
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

def square(a):
    return torch.pow(a, 2.)

class Policy(nn.Module):

    def __init__(self,
                 state_size=1,
                 action_size=1,
                 latent_size=0,
                 output_size=1,
                 hidden_size=64,
                 hidden_activation=F.tanh,
                 output_activation=None):
        super(Policy, self).__init__()

        self.input_size = state_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.h_activation = hidden_activation

        self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_mean = nn.Linear(self.hidden_size, self.output_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, self.output_size))


    def forward(self, x, old=False):
        x = self.h_activation(self.affine1(x))
        x = self.h_activation(self.affine2(x))

        action_mean = self.action_mean(x)
        if self.output_activation == 'sigmoid':
            action_mean = F.sigmoid(self.action_mean(x))
        elif self.output_activation == 'tanh':
            action_mean = F.tanh(self.action_mean(x))

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_log_std = torch.clamp(action_log_std, -5, 5)

        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

class DiscretePolicy(nn.Module):
    def __init__(self,
                 state_size=1,
                 action_size=1,
                 latent_size=0,
                 output_size=1,
                 hidden_size=64,
                 hidden_activation=F.tanh,
                 output_activation=None):
        super(DiscretePolicy, self).__init__()

        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_activation = output_activation

        self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_mean = nn.Linear(self.hidden_size, self.output_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.h_activation = hidden_activation


    def forward(self, x):
        x = self.h_activation(self.affine1(x))
        x = self.h_activation(self.affine2(x))
        x = x + torch.tensor([1e-10], requires_grad=True).type(x.type())
        action = F.softmax(self.action_mean(x), dim=1)

        return action

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial()
        return action.data

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.unsqueeze(1)))


class Posterior(nn.Module):

    def __init__(self, state_size, action_size, latent_size, hidden_size,
                 output_size=1):
        super(Posterior, self).__init__()

        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.affine31 = nn.Linear(self.hidden_size, output_size)
        self.affine32 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        h1 = F.relu(self.affine1(x))
        h2 = F.relu(self.affine2(h1))

        return self.affine31(h2), self.affine32(h2)

class DiscretePosterior(nn.Module):
    def __init__(self,
                 state_size=1,
                 action_size=1,
                 latent_size=1,
                 hidden_size=1,
                 output_size=1):
        super(DiscretePosterior, self).__init__()

        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        h1 = F.relu(self.affine1(x))
        h2 = F.relu(self.affine2(h1))

        return self.output(h2)

class DiscreteLSTMPosterior(nn.Module):
    def __init__(self,
                 state_size=1,
                 action_size=1,
                 latent_size=1,
                 hidden_size=1,
                 output_size=1):
        super(DiscreteLSTMPosterior, self).__init__()
        self.input_size = state_size # s+ c =12
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(hidden_size=self.hidden_size, input_size=self.input_size, batch_first=True, num_layers=2) #N, L , H_in
        # self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        # self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        # x = (B, L, D_in) = (1, history, 12)
        # pdb.set_trace()
        output, (h_n, c_n) = self.encoder(x) # h, c shape as (2, 1, hidden)
        return self.output(c_n[-1]) # 2 for 2 layers

class Value(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(state_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        # x = F.relu(self.affine1(x))
        # x = F.relu(self.affine2(x))
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

class Reward(nn.Module):
    def __init__(self, state_size, action_size, latent_size, hidden_size=100):
        super(Reward, self).__init__()

        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.affine1 = nn.Linear(self.input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        #self.reward_head.weight.data.mul_(0.1)
        #self.reward_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        rewards = F.sigmoid(self.reward_head(x))
        return rewards

class Transition(nn.Module):
    def __init__(self, latent_size, hidden_size=64):
        super(Transition, self).__init__()

        self.input_size = latent_size
        self.hidden_size = hidden_size
        self.affine1 = nn.Linear(self.input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, latent_size)


    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)

class TransitionLSTM(nn.Module):
    def __init__(self, latent_size, hidden_size=64, state_size=2):
        super(Transition, self).__init__()

        self.input_size = latent_size + state_size # 10 + 2
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(hidden_size=self.hidden_size, input_size=self.input_size, batch_first=True, num_layers=2) #N, L , H_in

    def forward(self, x):
        output, (h_n, c_n) = self.encoder(x) # h, c shape as (2, 1, hidden), 2 for 2 layers
        return self.output(output[-1]) # return (h_n, c_n) for last timestep
        # return self.affine2(x)
