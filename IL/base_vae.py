
from base_models import Policy, Posterior, DiscretePosterior, DiscreteLSTMPosterior
import torch.nn as nn
import torch
from torch.autograd import Variable
from bcolors import bcolors
import torch.nn.functional as F
import math

class VAE(nn.Module):
    def __init__(self,
                 policy_state_size=1, posterior_state_size=1,
                 policy_action_size=1, posterior_action_size=1,
                 policy_latent_size=1, posterior_latent_size=1,
                 posterior_goal_size=1,
                 policy_output_size=1,
                 history_size=1,
                 hidden_size=64,
                 use_goal_in_policy=True,
                 use_separate_goal_policy=True,
                 use_history_in_policy=False, args=None):
        '''
        state_size: State size
        latent_size: Size of 'c' variable
        goal_size: Number of goals.
        output_size:
        '''
        super(VAE, self).__init__()

        self.args = args
        self.history_size = history_size
        self.policy_state_size = policy_state_size
        self.posterior_latent_size = posterior_latent_size
        self.posterior_goal_size = posterior_goal_size
        self.use_goal_in_policy = use_goal_in_policy
        self.use_separate_goal_policy = use_separate_goal_policy
        self.use_history_in_policy = use_history_in_policy

        self.policy_latent_size = policy_latent_size
        if use_goal_in_policy:
            self.policy_latent_size += posterior_goal_size

        #if args.discrete:
        #    output_activation='sigmoid'
        #else:
        output_activation=None

        if use_history_in_policy:
            policy1_state_size = policy_state_size * history_size
        else:
            policy1_state_size = policy_state_size

        self.policy = Policy(state_size=policy1_state_size,
                             action_size=policy_action_size,
                             latent_size=self.policy_latent_size,
                             output_size=policy_output_size,
                             hidden_size=hidden_size,
                             output_activation=output_activation)

        if use_separate_goal_policy:
            self.policy_goal = Policy(
                    state_size=policy_state_size*self.history_size,
                    action_size=policy_action_size,
                    latent_size=posterior_goal_size,
                    output_size=policy_output_size,
                    hidden_size=hidden_size,
                    output_activation=output_activation)

        self.posterior = Posterior( # input (s, a, c)
                state_size=posterior_state_size*self.history_size,
                action_size=posterior_action_size,
                latent_size=posterior_latent_size+posterior_goal_size,
                output_size=posterior_latent_size,
                hidden_size=hidden_size)


    def encode(self, x, c):
        return self.posterior(torch.cat((x, c), 1))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode_goal_policy(self, x, g):
        action_mean, _, _ = self.policy_goal(torch.cat((x, g), 1))
        if 'circle' in self.args.env_type:
            action_mean = action_mean / torch.norm(action_mean, dim=1).unsqueeze(1)
        return action_mean

    def decode(self, x, c):
        action_mean, action_log_std, action_std = self.policy(
                torch.cat((x, c), 1))
        if 'circle' in self.args.env_type:
            action_mean = action_mean / torch.norm(action_mean, dim=1).unsqueeze(1)

        return action_mean

    def forward(self, x, c, g, only_goal_policy=False):
        if only_goal_policy: # False
            decoder_output_2 = self.decode_goal_policy(x, g)
            # Return a tuple as the else part below. Caller should expect a
            # tuple always.
            return decoder_output_2,
        else:
            mu, logvar = self.encode(x, c)
            c[:,-self.posterior_latent_size:] = self.reparameterize(mu, logvar)

            decoder_output_1 = None
            decoder_output_2 = None


        if self.use_goal_in_policy: # False
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(x, c)
            else:
                decoder_output_1 = self.decode(
                        x[:, -self.policy_state_size:], c)
        else:
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(
                        x, c[:,-self.posterior_latent_size:])
            else:
                decoder_output_1 = self.decode(
                        x[:, -self.policy_state_size:],
                        c[:,-self.posterior_latent_size:])

            if self.use_separate_goal_policy:
                decoder_output_2 = self.decode_goal_policy(x, g)

            '''
            decoder_output1 is action of policy 
            decoder_output2 is action goal_policy
            mu, logvar is posterior.
            '''
            return decoder_output_1, decoder_output_2, mu, logvar

class DiscreteVAE(VAE):
    def __init__(self, temperature=5.0, **kwargs):
        '''
        state_size: State size
        latent_size: Size of 'c' variable
        goal_size: Number of goals.
        output_size:
        '''
        super(DiscreteVAE, self).__init__(**kwargs)
        if kwargs['use_boundary']:
            self.posterior = DiscreteLSTMPosterior(
                    state_size=kwargs['posterior_state_size'],
                    action_size=kwargs['posterior_action_size'],
                    latent_size=kwargs['posterior_latent_size']+kwargs['posterior_goal_size'],
                    output_size=kwargs['posterior_latent_size'],
                    hidden_size=kwargs['hidden_size'],
            )
        else:
            self.posterior = DiscretePosterior(
                state_size=kwargs['posterior_state_size']*self.history_size,
                action_size=kwargs['posterior_action_size'],
                latent_size=kwargs['posterior_latent_size']+kwargs['posterior_goal_size'],
                output_size=kwargs['posterior_latent_size'],
                hidden_size=kwargs['hidden_size'],
        )
        self.encoder_softmax = nn.Softmax(dim=1)
        self.temperature = temperature
        self.init_temperature = temperature
        print(bcolors.Blue+"Initial temperature: {}".format(temperature)+
                bcolors.Endc)

    def update_temperature(self, epoch):
        '''Update temperature.'''
        r = 5e-4  # will become 1.0 after 3000 epochs 
        # r = 33e-4 will become 1.0 after 500 epochs and 0.18 after 1000 epochs.
        # r = 0.023 # Will become 0.1 after 100 epochs if initial temp is 1.0
        # r = 0.011 # Will become 0.1 after 200 epochs if initial temp is 1.0
        self.temperature = max(0.1, self.init_temperature * math.exp(-r*epoch))


    def encode(self, x, c):
        '''Return the log probability output for the encoder.'''
        logits = self.posterior(torch.cat((x, c), 1))
        return logits

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dtype = logits.data.type()
        y = logits + Variable(self.sample_gumbel(logits.size())).type(dtype)
        y = F.softmax(y / temperature, dim=1)
        # shape = y.size()
        # _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        # return (y_hard - y).detach() + y
        return y

    def reparameterize(self, logits, temperature, eps=1e-10):
        if self.training:
            probs = self.gumbel_softmax_sample(logits, temperature)
        else:
            probs = F.softmax(logits / temperature, dim=1)
        return probs

    def forward(self, x, c, g, only_goal_policy=False):
        # if only_goal_policy:
        #     decoder_output_2 = self.decode_goal_policy(x, g)
        #     # Return a tuple as the else part below. Caller should expect a
        #     # tuple always.
        #     return decoder_output_2,

        c_logits = self.encode(x, c)
        c[:, -self.posterior_latent_size:] = self.reparameterize(
                c_logits, self.temperature)

        decoder_output_1 = None
        decoder_output_2 = None

        # if self.use_goal_in_policy:
        #     if self.use_history_in_policy:
        #         decoder_output_1 = self.decode(x, c)
        #     else:
        #         decoder_output_1 = self.decode(x[:,-self.policy_state_size:], c)
        # else:
        if self.use_history_in_policy:
            decoder_output_1 = self.decode(
                    x, c[:,-self.posterior_latent_size:])
        else:
            decoder_output_1 = self.decode(
                    x[:, -self.policy_state_size:],
                    c[:,-self.posterior_latent_size:])

        # if self.use_separate_goal_policy:
        #     decoder_output_2 = self.decode_goal_policy(x, g)

        '''
        decoder_output_1 is action
        decoder_output_2 is goal_conditioned_action
        c_logits is encoded variable (posterior)
        '''
        return decoder_output_1, None, c_logits

