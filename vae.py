import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from load_expert_traj import Expert
from grid_world import State, Action, TransitionFunction, RewardFunction, RewardFunction_SR2
from grid_world import create_obstacles, obstacle_movement, sample_start
from itertools import product
from models import Policy, Posterior

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--expert-path', default="L_expert_trajectories/", metavar='G',
                    help='path to the expert trajectory files')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#-----Environment-----#
width = height = 21
obstacles = create_obstacles(width, height, 'diverse')

set_diff = list(set(product(tuple(range(7,13)),tuple(range(7,13)))) - set(obstacles))
state = State(sample_start(set_diff), obstacles)

T = TransitionFunction(width, height, obstacle_movement)

#if args.expert_path == 'SR2_expert_trajectories/':
#    R = RewardFunction_SR2(-1.0,1.0,width)
#else:
#    R = RewardFunction(-1.0,1.0)


class VAE(nn.Module):
    def __init__(self, state_size, action_size, latent_size, output_size, hidden_size):
        super(VAE, self).__init__()

        self.history_size = 4
        self.policy = Policy(state_size=state_size*self.history_size,
                            action_size=action_size, 
                            latent_size=latent_size, 
                            output_size=output_size, 
                            hidden_size=hidden_size, 
                            output_activation='sigmoid')

        self.posterior = Posterior(state_size=state_size*self.history_size, 
                                   action_size=action_size, 
                                   latent_size=latent_size, 
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

    def decode(self, x, c):
        action_mean, action_log_std, action_std = self.policy(torch.cat((x, c), 1))
        return action_mean

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        c[:,0] = self.reparameterize(mu, logvar)
        return self.decode(x, c), mu, logvar


model = VAE(state.state.shape[0], 0, 2, 8, 64)

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    #MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #KLD = 0.5 * torch.sum(mu.pow(2))

    return BCE + KLD
    #return MSE + KLD


def train(epoch, expert, Transition):
    model.train()
    history_size = model.history_size
    train_loss = 0
    for batch_idx in range(10): # 10 batches per epoch
        batch = expert.sample(args.batch_size)
        print len(batch.state[0])
        print len(batch.state[1])
        x_data = torch.Tensor(batch.state)
        N = x_data.size(1)
        x = -1*torch.ones(x_data.size(0), history_size, x_data.size(2))
        x[:,(history_size-1),:] = x_data[:,0,:]

        a = Variable(torch.Tensor(batch.action))

        _, c2 = torch.Tensor(batch.c).max(2)
        c2 = c2.float()[:,0].unsqueeze(1)
        c1 = -1*torch.ones(c2.size())
        c = torch.cat((c1,c2),1)

        #c_t0 = Variable(c[:,0].clone().view(c.size(0), 1))

        if args.cuda:
            a = a.cuda()
            #c_t0 = c_t0.cuda()

        optimizer.zero_grad()
        for t in range(N):
            #x_t0 = Variable(x[:,0,:].clone().view(x.size(0), x.size(2)))
            #x_t1 = Variable(x[:,1,:].clone().view(x.size(0), x.size(2)))
            #x_t2 = Variable(x[:,2,:].clone().view(x.size(0), x.size(2)))
            #x_t3 = Variable(x[:,3,:].clone().view(x.size(0), x.size(2)))
            input_x = Variable(x[:,:,:].view(x.size(0), history_size*x.size(2)).clone())
            c_t0 = Variable(c)

            if args.cuda:
                input_x = input_x.cuda()
                #x_t0 = x_t0.cuda()
                #x_t1 = x_t1.cuda()
                #x_t2 = x_t2.cuda()
                #x_t3 = x_t3.cuda()
                c_t0 = c_t0.cuda()


            recon_batch, mu, logvar = model(input_x, c_t0)
            loss = loss_function(recon_batch, a[:,t,:], mu, logvar)
            loss.backward()
            train_loss += loss.data[0]

            pred_actions = recon_batch.data.cpu().numpy()

            x[:,:3,:] = x[:,1:,:]
            # get next state and update x
            for b_id in range(pred_actions.shape[0]):
                action = Action(np.argmax(pred_actions[b_id,:]))
                state = State(x[b_id,3,:].cpu().numpy(), obstacles)
                next_state = Transition(state, action, 0)
                x[b_id,3,:] = torch.Tensor(next_state.state)

            # update c
            c[:,0] = model.reparameterize(mu, logvar).data.cpu()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, 200.0,
                100. * batch_idx / 20.0,
                loss.data[0] / args.batch_size))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / 200.0))


def test(Transition):
    model.eval()
    history_size = model.history_size
    #test_loss = 0

    for _ in range(20):
        c = expert.sample_c()
        N = c.shape[0]
        c = np.argmax(c[0,:])
        #if args.expert_path == 'SR_expert_trajectories/':
        #    if c == 1:
        #        half = 0
        #    elif c == 3:
        #        half = 1
        #elif args.expert_path == 'SR2_expert_trajectories/':
        #    half = c
        #if args.expert_path == 'SR_expert_trajectories/' or args.expert_path == 'SR2_expert_trajectories/':
        #    if half == 0: # left half
        #        set_diff = list(set(product(tuple(range(0, (width/2)-3)), tuple(range(1, height)))) - set(obstacles))
        #    elif half == 1: # right half
        #        set_diff = list(set(product(tuple(range(width/2, width-2)), tuple(range(2, height)))) - set(obstacles))
        #else:
        #    set_diff = list(set(product(tuple(range(3, width-3)), repeat=2)) - set(obstacles))
        set_diff = list(set(product(tuple(range(7,13)),tuple(range(7,13)))) - set(obstacles))

        start_loc = sample_start(set_diff)
        s = State(start_loc, obstacles)
        R.reset()
        c = torch.from_numpy(np.array([-1.0,c])).unsqueeze(0).float()

        print 'c is ', c[0,1]

        c = Variable(c)

        x = -1*torch.ones(1, history_size, 2)

        if args.cuda:
            x = x.cuda()
            c = c.cuda()

        for t in range(N):

            x[:,:(history_size-1),:] = x[:,1:,:]
            curr_x = torch.from_numpy(s.state).unsqueeze(0)
            if args.cuda:
                curr_x = curr_x.cuda()

            x[:,(history_size-1),:] = curr_x

            #x_t0 = Variable(x[:,0,:])
            #x_t1 = Variable(x[:,1,:])
            #x_t2 = Variable(x[:,2,:])
            #x_t3 = Variable(x[:,3,:])

            input_x = Variable(x[:,:,:].view(x.size(0), history_size*x.size(2)).clone())

            mu, logvar = model.encode(input_x, c)
            c[:,0] = model.reparameterize(mu, logvar)
            pred_a = model.decode(input_x, c).data.cpu().numpy()
            pred_a = np.argmax(pred_a)
            print pred_a
            next_s = Transition(s, Action(pred_a), R.t)

            s = next_s

            #test_loss += loss_function(recon_batch, data, mu, logvar).data[0]


    #test_loss /= len(test_loader.dataset)
    #print('====> Test set loss: {:.4f}'.format(test_loss))


expert = Expert(args.expert_path, 2)
expert.push()

for epoch in range(1, args.epochs + 1):
    train(epoch, expert, T)
    #test(epoch)
    #sample = Variable(torch.randn(64, 20))
    #if args.cuda:
    #    sample = sample.cuda()
    #sample = model.decode(sample).cpu()
    #save_image(sample.data.view(64, 1, 28, 28),
    #           'results/sample_' + str(epoch) + '.png')

test(T)
