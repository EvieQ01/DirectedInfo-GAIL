import torch
from copy import copy, deepcopy
import random
import numpy as np
import argparse
import h5py
import os
import pdb
import math

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import sys

from models import Encoder
sys.path.append('../IL')
from boundary_utils import get_boundary_from_all_traj
import grid_world as gw
from load_expert_traj import Expert, ExpertHDF5

# from utils.logger import Logger, TensorboardXLogger
from dataset_utils import sub_traj_dataset, noise_sample
import time
from tqdm import trange
from models import Generator, Discriminator, QHead
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from plot_utils import plot_trajectory
import seaborn as sns
import wandb
def main(args):

    # Create Logger
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))

    device = 'cpu'
    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor
        device = 'cuda'


    label_dtype = torch.LongTensor
    if args.cuda:
        label_dtype = torch.cuda.LongTensor

    expert = ExpertHDF5(args.expert_path, args.vae_state_size)
    expert.push(only_coordinates_in_state=True, one_hot_action=True)
    traj_expert = expert.sample_all()
    state_expert, action_expert, c_expert, _ = traj_expert
    boundary_list = get_boundary_from_all_traj(state_expert)  # (50, 15, 2)   
    dataset = sub_traj_dataset(state_expert, boundary_list=boundary_list, max_len=args.max_len, batch_size=args.batch_size,set_diff=expert.set_diff)
    
    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ = nn.CrossEntropyLoss()

    # Initialise the network.
    # netG = Generator(input_size=args.c_dim+args.z_dim+len(expert.set_diff), max_len=args.max_len, output_size=len(expert.set_diff)).type(dtype)
    netG = Generator(embed_size=args.embed_size).type(dtype)
    print(netG)

    # embed_x  -> 1
    discriminator = Discriminator(input_size=args.embed_size).type(dtype)
    print(discriminator)

    # x  -> c
    netQ = QHead(input_size=args.embed_size, output_size=args.c_dim).type(dtype)
    print(netQ)

    # , {'params': encoder.s_embed.parameters()}
    encoder = Encoder(vocab_size=len(expert.set_diff) + 1, embed_size=args.embed_size, c_size=args.c_dim).type(dtype)
    optimD = optim.Adam([{'params': discriminator.parameters()}], lr=args.lr)#, weight_decay=1e-3)#, betas=(args.beta1, args.beta2))
    optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=args.lr)#, weight_decay=1e-3)#, betas=(args.beta1, args.beta2))
    real_label = 1
    fake_label = 0
    for epoch in trange(args.num_epochs):
        # training
        for iter in range(dataset.size // args.batch_size):
            batch, boundary_index = dataset.sample_padded_batch() # (B, len)
            noise_z, noise_c = noise_sample(dis_c_dim=args.c_dim, z_dim=len(expert.set_diff), batch_size=args.batch_size, dtype=dtype) # (B, c+z)
            batch = batch.to(device) # (B, len)
            batch_feat = encoder.s_embed(batch.type(label_dtype))
            # Updating discriminator and DHead
            optimD.zero_grad()

            # Fake data
            label = torch.full((args.batch_size, ), fake_label).type(dtype=dtype)
            feat_z, feat_c = encoder(noise_z.type(label_dtype), noise_c.type(label_dtype)) # (B)(B) -> (B, embed) (B, embed)
            fake_data_feat = netG(noise_z=feat_z, context=feat_c, teacher_force=args.teacher_force, x=batch_feat) #TODO (B, max_len, dim) [packed]
            #TODO mask
            probs_fake = torch.sigmoid(discriminator(fake_data_feat.detach())).view(-1)
            loss_fake = criterionD(probs_fake, label)
            # Calculate gradients.
            loss_fake.backward(retain_graph=True)

            # Real data
            label.fill_(real_label)
            probs_real = torch.sigmoid(discriminator(batch_feat)).view(-1) # discri(B, hidden) -> (B, 1)
            loss_real = criterionD(probs_real, label)
            # Calculate gradients.
            loss_real.backward(retain_graph=True)

            # Net Loss for the discriminator
            D_loss = loss_real + loss_fake # probs are before softmax
            # Update parameters
            optimD.step()

            # Updating Generator and QHead
            optimG.zero_grad()
            # Fake data treated as real.
            probs_fake = torch.sigmoid(discriminator(fake_data_feat)).view(-1)
            label.fill_(real_label)
            gen_loss = criterionD(probs_fake, label)

            q_logits = netQ(fake_data_feat) # change this from origin infoGAN
            target = torch.tensor(noise_c).type(dtype=label_dtype)
            # Calculating loss for discrete latent code.
            #(B, c_dim), (B, c_dim)
            # pdb.set_trace()
            posterior_loss = criterionQ(q_logits, target.detach())

            # [extra distance constraints on c]
            view2 = torch.repeat_interleave(boundary_index[:, 0], args.batch_size) # B^2
            view1 = boundary_index[:, 1].repeat(args.batch_size) # B^2
            mask = torch.bitwise_or(torch.bitwise_and((view1 == view2 ).reshape((args.batch_size, -1)), boundary_index[:, 0] != -1), \
                    torch.bitwise_and((view1 == view2 ).reshape((args.batch_size, -1)), boundary_index[:, 1] != -1))
            mask = mask + mask.T
            
            q_logits_detach = netQ(fake_data_feat.detach()) 
            dist = torch.softmax(q_logits_detach, dim=-1).unsqueeze(0) - torch.softmax(q_logits_detach, dim=-1).unsqueeze(1) # B, B , dim_c
            # dist = q_logits_detach.unsqueeze(0) - q_logits_detach.unsqueeze(1) # B, B , dim_c
            dist_loss = torch.sqrt(torch.mean(torch.sum(dist*dist, dim=-1)* mask.type(dtype) / 2 ) ) 
            # Net loss for generator.
            G_loss = gen_loss * args.lambda_gen + posterior_loss * args.lambda_post + 1 / (dist_loss + 1e-5) * args.lambda_dist
            # Calculate gradients.
            # pdb.set_trace()
            G_loss.backward()
            # if epoch == 100:
            #     pdb.set_trace()
            # Update parameters.
            # torch.nn.utils.clip_grad.clip_grad_norm_(netG.parameters(), max_norm=5.)
            # torch.nn.utils.clip_grad.clip_grad_norm_(discriminator.parameters(), max_norm=5.)
            # torch.nn.utils.clip_grad.clip_grad_norm_(netQ.parameters(), max_norm=5.)
            optimG.step()
            # pdb.set_trace()
            # Check progress of training.
            if iter % 10 == 0:
                print('epoch[%d/%d]\titer[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f + %.4f * lambda + %.4f'
                    % (epoch+1, args.num_epochs, iter, dataset.size // args.batch_size, 
                        D_loss.item(), gen_loss.item(), posterior_loss.item(), dist_loss.item()))
                if args.wandb:
                    train_dict = {'debug/D_loss': D_loss.item(),
                                    'debug/G_loss': G_loss.item(), 
                                    'debug/gen_loss': gen_loss.item(), 
                                    'debug/posterior_loss': posterior_loss.item(), }
                    wandb.log(train_dict)

        # testing
        # dict of state and pred context
        if (epoch  )% 10 == 0:
            state_and_pred_context = test_for_context(Qnet=netQ, encoder=encoder.s_embed, expert=expert, boundary_list=boundary_list,\
                                                        dataset=dataset, dtype=dtype, label_dtype=label_dtype,  num_test_samples=10)

            # plot
            fig_dir = os.path.join(os.path.dirname(args.results_pkl_path), 'visualize')
            os.makedirs(fig_dir,exist_ok='True')
            
            wandb_log_stamp = now if args.wandb else False
            plot_trajectory(traj_context_data=state_and_pred_context,
                            grid_size=(15, 11),
                            color_map=sns.color_palette("Blues_r"),
                            figsize=(6, 6),
                            obstacles=expert.obstacles,
                            save_path=fig_dir, wandb_log_stamp=wandb_log_stamp, eposode=epoch)

def test_for_context(Qnet:nn.Module, encoder:nn.Module, expert, boundary_list, dataset, dtype, label_dtype, num_test_samples=10):
    Qnet.eval()
    results = {'true_traj_state': [],'pred_context': []}
    # get sub_trajs for each traj
    for i_traj in range(num_test_samples):
        batch = expert.sample_index(ind=i_traj)
        ep_state, ep_action, _, ep_mask = batch
        ep_state = np.array(ep_state, dtype=np.float32)
        # ep_action will be (N, A)
        episode_len = ep_state.shape[1]
        boundary_time_stamp_list = []
        for history_t in range(episode_len):
            xy_posi = (ep_state[0][history_t][0], ep_state[0][history_t][1])
            if xy_posi in boundary_list:
                # print('boundary: ', xy_posi)
                if history_t not in boundary_time_stamp_list:
                    boundary_time_stamp_list.append(history_t)

        pred_context = []
        true_traj = []
        # pred_traj = []
        sub_traj_index = 0
        cross_point_index = 0
        for t in range(ep_state.shape[1]):
            x_var = Variable(torch.from_numpy(ep_state[:, cross_point_index:t+1, :]).type(dtype))
            # TODO all c shift for 1 timestep
            if t in boundary_time_stamp_list or t == 0:
                # update c and switch subtask
                sub_traj_index += 1
                if sub_traj_index > 1: # only when sub_traj_index ==2, begin to change sub_traj_id, skip the begining
                    cross_point_index = boundary_time_stamp_list[sub_traj_index - 2] + 1
                    if args.debug:
                        print(cross_point_index, 'is cross_point_index')
                # get posterior c' = Q(x_{1:t}, c-1)
                # store latent variables (logits or mu)
                if t != 0:
                    x_feat = encoder(dataset.turn_xy_posi_to_onehot(x_var[0]).unsqueeze(0).type(label_dtype))
                    logits_q = F.softmax(Qnet(x_feat), dim=-1)[0] #(B, c)
                    for _ in range(x_var.shape[-2]):
                        pred_context.append(logits_q.data.cpu().numpy())
                    if args.debug:
                        print(x_var)
            # Store the "true" state
            true_traj.append(ep_state[0, t, :])
        if len(pred_context) < ep_state.shape[-2]:
            # only add this when last_time_step is not a boundary
            # pdb.set_trace()
            for _ in range(x_var.shape[-2]):
                pred_context.append(logits_q.data.cpu().numpy())
        true_traj_state_arr = np.array(true_traj)

        results['true_traj_state'].append(true_traj_state_arr)
        results['pred_context'].append(np.array(pred_context))
    Qnet.train()
    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VAE Example')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--max_len', type=int, default=5, metavar='N',
                        help='max length of sub sequence')
    parser.add_argument('--num_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--checkpoint_every_epoch', type=int, default=10,
                        help='Save models after ever N epochs.')
    # Run on GPU
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Disable CUDA training')
    parser.set_defaults(cuda=False)

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging ' \
                              'training status')
    parser.add_argument('--expert_path', default='../IL/h5_trajs/room_trajs/traj_len_16',
                        metavar='G',
                        help='path to the expert trajectory files')

    parser.add_argument('--vae_state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--vae_action_size', type=int, default=4,
                        help='Action size for VAE.')
    parser.add_argument('--c_dim', type=int, default=10,
                        help='Context size for VAE.')
    parser.add_argument('--embed_size', type=int, default=10,
                        help='noise size for generator.')

    # lr
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--lambda_post', type=float, default=.001, help='param for poset')
    parser.add_argument('--lambda_dist', type=float, default=.001, help='param for dist')
    parser.add_argument('--lambda_gen', type=float, default=1., help='param for gen')


    # Use boundary
    parser.add_argument('--use_boundary', action='store_true')
    parser.set_defaults(use_boundary=True)


    # Results dir
    parser.add_argument('--results_dir', type=str, default='./results/vae/room_traj/discrete/traj_room_centre_len_16',
                        help='Directory to save final results in.')
    # Checkpoint directory to load pre-trained models.
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint path to load pre-trained models.')
    parser.add_argument('--finetune_path', type=str, default='',
                        help='pre-trained models to finetune.')
    parser.add_argument('--results_pkl_path', default='./results/test_predict/room_traj/context_4/pred_result_cp_1000.pth')

    # training hyperparams
    
    # debug
    parser.add_argument('--debug', action='store_true', help='whether print out debugging message')
    parser.add_argument('--wandb', action='store_true', help='whether save on wandb')
    parser.add_argument('--teacher_force', action='store_true', help='whether use true state in RNN generator')
    args = parser.parse_args()
    now = time.strftime(f"Gen{args.lambda_gen}_Post{args.lambda_post}_Dist{args.lambda_dist}"+"%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    
    if args.wandb:
        wandb.init(project="infoGAN_grid_room", entity="evieq01")
        wandb.run.name = f'{now}'
        wandb.config.update(args)

    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global_args = args
    main(args)
