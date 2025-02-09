
import pdb
from base_models import TransitionLSTM
from base_models import Policy, Posterior, DiscretePosterior, DiscreteLSTMPosterior, Transition, DiscretePolicy
import torch.nn as nn
import torch
from torch.autograd import Variable
from bcolors import bcolors
import torch.nn.functional as F
import math

class VAE(nn.Module):
    def __init__(self,
                 policy_state_size=1, posterior_state_size=1,
                 dynamics_state_size=1,
                 policy_latent_size=1, posterior_latent_size=1,
                 policy_output_size=1,
                 hidden_size=64,
                use_boundary=False, args=None):
        '''
        state_size: State size
        latent_size: Size of 'c' variable
        goal_size: Number of goals.
        output_size:
        '''
        super(VAE, self).__init__()

        self.args = args
        self.policy_state_size = policy_state_size
        self.posterior_latent_size = posterior_latent_size
        self.use_boundary = use_boundary

        self.policy_latent_size = policy_latent_size

        #if args.discrete:
        #    output_activation='sigmoid'
        #else:

        # policy_class = DiscretePolicy if args.discrete_action else Policy
        # self.transition = nn.LSTMCell(posterior_latent_size, posterior_latent_size) #(10 -> 10)
    # def encode(self, x, c):
    #     return self.posterior(torch.cat((x, c), 1))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    # def decode_goal_policy(self, x, g):
    #     action_mean, _, _ = self.policy_goal(torch.cat((x, g), 1))
    #     if 'circle' in self.args.env_type:
    #         action_mean = action_mean / torch.norm(action_mean, dim=1).unsqueeze(1)
    #     return action_mean

    def decode(self, x, c):

        # pdb.set_trace()
        action_mean, action_log_std, action_std = self.policy(torch.cat((x, c), -1))

        return action_mean

    def forward(self, x, c, g):
        mu, logvar = self.encode(x, c)
        c[:,-self.posterior_latent_size:] = self.reparameterize(mu, logvar)

        decoder_output_1 = None
        decoder_output_2 = None

        decoder_output_1 = self.decode(
                    x[:, :, -self.policy_state_size:],
                    c[:, :, -self.posterior_latent_size:])


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
        self.posterior = DiscreteLSTMPosterior(
                input_size=kwargs['posterior_state_size'] + kwargs['posterior_latent_size'],
                output_size=kwargs['posterior_latent_size'],
                hidden_size=kwargs['hidden_size'],
        )
        # (\tau_j | c_j-1, c_j)
        self.Q_alpha =  DiscretePosterior(
                input_size=2 * kwargs['posterior_latent_size'] + kwargs['posterior_state_size'] + kwargs['policy_output_size'], 
                output_size=kwargs['dynamics_state_size'],
                hidden_size=kwargs['hidden_size'],
        )

        # (\tau_j |c_j)
        self.Q_gamma =  DiscretePosterior(
                input_size=kwargs['posterior_latent_size'] + kwargs['posterior_state_size']+ kwargs['policy_output_size'],
                output_size=kwargs['dynamics_state_size'],
                hidden_size=kwargs['hidden_size'],
        )
        policy_class = DiscretePolicy
        self.policy = policy_class(input_size=kwargs['posterior_latent_size'] + kwargs['posterior_state_size'],
                             output_size=kwargs['policy_output_size'],
                             hidden_size=kwargs['hidden_size'])

        # if args.use_lstm_transition:
        #     self.transition = TransitionLSTM(latent_size=posterior_latent_size, state_size=posterior_state_size) #(10 + 2-> 10)
        # else:
        self.transition = Transition(latent_size=kwargs['posterior_latent_size']) #(10 -> 10)
        self.encoder_softmax = nn.Softmax(dim=1)
        self.temperature = temperature
        self.init_temperature = temperature
        print(bcolors.Blue+"Initial temperature: {}".format(temperature)+
                bcolors.Endc)

    def update_temperature(self, epoch):
        '''Update temperature.'''
        # r = 5e-4  # will become 1.0 after 3000 epochs 
        # r = 33e-4 will become 1.0 after 500 epochs and 0.18 after 1000 epochs.
        # r = 0.023 # Will become 0.1 after 100 epochs if initial temp is 1.0
        r = 0.011 # Will become 0.1 after 200 epochs if initial temp is 1.0
        self.temperature = max(0.1, self.init_temperature * math.exp(-r*epoch))


    def encode(self, x, c):
        '''Return the log probability output for the encoder.'''
        # x = (B, history, 2)
        # c = (B, 10)
        # x_c_history 
        # pdb.set_trace()
        duplicate_c = c.repeat(1, x.shape[-2], 1)
        logits = self.posterior(torch.cat((x, duplicate_c), -1))
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

    def forward(self, x, c,):

        c_logits = self.encode(x, c)
        # remake c of last timestep
        # pdb.set_trace()
        c_next = self.reparameterize(c_logits, self.temperature) # c_next

        decoder_output_1 = None

        # Use last timestep \pi(x_t, c)
        decoder_output_1 = self.decode(x[:,-1, :], c_next) # (B, 2) (B, 10)
        '''
        decoder_output_1 is action
        c_logits is encoded variable (posterior)
        '''
        return decoder_output_1, c_logits
    
