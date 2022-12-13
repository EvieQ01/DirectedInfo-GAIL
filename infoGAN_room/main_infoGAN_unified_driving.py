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
from manipulate_dataset import update_dataset

from models import Encoder
from utils.boundary_utils_continuous import get_div_from_all_traj_continuous
from utils.load_expert_traj import CircleExpertHDF5


# from utils.logger import Logger, TensorboardXLogger
from utils.dataset_utils_driving import dynamic_sub_traj_dataset, noise_sample
import time
from tqdm import trange
from models import Generator, Discriminator, QHead
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from utils.plot_utils import plot_trajectory_circle
import seaborn as sns
import wandb
import matplotlib as plt

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

    expert = CircleExpertHDF5(args.expert_path, args.state_size)
    expert.push(only_coordinates_in_state=False, one_hot_action=True)
    traj_expert = expert.sample_all()
    state_expert, action_expert, c_expert, _ = traj_expert
    divergence = get_div_from_all_traj_continuous(state_expert, delta_t=5, neighbor_k=10) # (50, 15, 2)   
    # dataset = sub_traj_dataset(state_expert, boundary_list=boundary_list, max_len=args.max_len, batch_size=args.batch_size,set_diff=expert.set_diff)
    dataset = dynamic_sub_traj_dataset(all_traj=list(state_expert), max_len=args.max_len,\
         batch_size=args.batch_size,div=divergence)
    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ = nn.CrossEntropyLoss()

    # Initialise the network.
    # netG = Generator(input_size=args.c_dim+args.z_dim+len(expert.set_diff), max_len=args.max_len, output_size=len(expert.set_diff)).type(dtype)
    netG = Generator(embed_size=args.embed_size).type(dtype)
    print(netG)

    # embed_x  -> 1
    discriminator = Discriminator(input_size=args.state_size).type(dtype)
    print(discriminator)

    # x  -> c
    netQ = QHead(input_size=args.state_size, output_size=args.c_dim).type(dtype)
    print(netQ)

    # , {'params': encoder.s_embed.parameters()}
    encoder = Encoder(embed_size=args.embed_size, c_size=args.c_dim).type(dtype)
    
    optimizer = optim.RMSprop # Adam RMSprop
    optimD = optimizer([{'params': discriminator.parameters()}], lr=args.lr)#, weight_decay=1e-3)#, betas=(args.beta1, args.beta2))
    optimG = optimizer([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=args.lr)#, weight_decay=1e-3)#, betas=(args.beta1, args.beta2))
    schedulerD = optim.lr_scheduler.StepLR(optimD, step_size=30, gamma=0.1)
    schedulerG = optim.lr_scheduler.StepLR(optimG, step_size=30, gamma=0.1)
    decay_ratio = 1
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 60
    # lr = 0.0005   if 60 <= epoch < 90
    
    # optimD = optim.Adam([{'params': discriminator.parameters()}], lr=args.lr)#, weight_decay=1e-3)#, betas=(args.beta1, args.beta2))
    # optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=args.lr)#, weight_decay=1e-3)#, betas=(args.beta1, args.beta2))
    real_label = 1
    fake_label = 0
    fix_dataset_flag = False
    for epoch in trange(args.num_epochs):
        # training
        for iter in range(dataset.cur_size // args.batch_size):
            # get sorted batch
            batch, potential_boundary, sorted_seq_lengths = dataset.sample_padded_batch_sorted() # (B, len)
            noise_z, noise_c = noise_sample(dis_c_dim=args.c_dim, z_dim=args.state_size, batch_size=args.batch_size, dtype=dtype) # (B, z), (B)
            batch = batch.to(device) # (B, len)
            
            batch_feat_packed = pack_padded_sequence(input=batch, lengths=sorted_seq_lengths, batch_first=True, enforce_sorted=True)
            # Updating discriminator and DHead
            optimD.zero_grad()

            # Fake data
            label = torch.full((args.batch_size, ), fake_label).type(dtype=dtype)
            feat_z, feat_c = noise_z.to(device), encoder.c_embed(noise_c.to(device))#.type(label_dtype) # (B)(B) -> (B, embed) (B, embed)
            # pdb.set_trace()
            fake_data_feat = netG(noise_z=feat_z, context=feat_c, teacher_force=args.teacher_force, x=batch) #TODO (B, max_len, dim) [packed]
            fake_data_feat_packed_detach = pack_padded_sequence(input=fake_data_feat.detach(), lengths=sorted_seq_lengths, batch_first=True, enforce_sorted=True)
            fake_data_feat_packed = pack_padded_sequence(input=fake_data_feat, lengths=sorted_seq_lengths, batch_first=True, enforce_sorted=True)
            #TODO mask
            probs_fake = torch.sigmoid(discriminator(fake_data_feat_packed_detach)).view(-1)
            loss_fake = criterionD(probs_fake, label)
            # Calculate gradients.
            loss_fake.backward(retain_graph=True)

            # Real data
            label.fill_(real_label)
            probs_real = torch.sigmoid(discriminator(batch_feat_packed)).view(-1) # discri(B, hidden) -> (B, 1)
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
            probs_fake = torch.sigmoid(discriminator(fake_data_feat_packed)).view(-1)
            label.fill_(real_label)
            gen_loss = criterionD(probs_fake, label)

            q_logits = netQ(fake_data_feat_packed) # change this from origin infoGAN
            target = torch.tensor(noise_c).type(dtype=label_dtype)
            # Calculating loss for discrete latent code.
            #(B, c_dim), (B, c_dim)
            # pdb.set_trace()
            posterior_loss = criterionQ(q_logits, target.detach())

            # [extra distance constraints on c]
            # view2 = torch.repeat_interleave(boundary_index[:, 0], args.batch_size) # B^2
            # view1 = boundary_index[:, 1].repeat(args.batch_size) # B^2
            # mask = torch.bitwise_or(torch.bitwise_and((view1 == view2 ).reshape((args.batch_size, -1)), boundary_index[:, 0] != -1), \
            #         torch.bitwise_and((view1 == view2 ).reshape((args.batch_size, -1)), boundary_index[:, 1] != -1))
            # mask = mask + mask.T
            
            q_logits_detach = netQ(fake_data_feat_packed_detach) 
            
            # adjascent sequences.
            # dist = torch.softmax(q_logits_detach, dim=-1).unsqueeze(0) - torch.softmax(q_logits_detach, dim=-1).unsqueeze(1) # B, B , dim_c
            dist_loss = torch.tensor(0)
            # adjacent_mask = dataset.get_adjacent_mask(potential_boundary=potential_boundary) # (B, B)
            
            # # pdb.set_trace()
            # if args.distance_type == 'l1':
            #     dist_loss = torch.sum(torch.mean(nn.L1Loss(reduction='none')(dist, torch.zeros_like(dist)),dim=-1) * adjacent_mask.type(dtype) )
            # elif args.distance_type == 'l2':
            #     dist_loss = torch.sum(torch.sum(dist*dist, dim=-1)* adjacent_mask.type(dtype) / 2 ) 

            # Net loss for generator.
            # [2] negative distance loss
            G_loss = gen_loss * args.lambda_gen + posterior_loss * args.lambda_post #+ dist_loss  * args.lambda_dist
            # [1] 1 / distance loss
            # G_loss = gen_loss * args.lambda_gen + posterior_loss * args.lambda_post + 1 / (dist_loss + 1e-5) * args.lambda_dist
            
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
                    % (epoch+1, args.num_epochs, iter, dataset.cur_size // args.batch_size, 
                        D_loss.item(), gen_loss.item(), posterior_loss.item(), dist_loss.item()))
                if args.wandb:
                    train_dict = {'debug/D_loss': D_loss.item(),
                                    'debug/G_loss': G_loss.item(), 
                                    'debug/gen_loss': gen_loss.item(), 
                                    'debug/posterior_loss': posterior_loss.item(),
                                    'debug/distance_loss': dist_loss.item(), }
                    wandb.log(train_dict)

        schedulerD.step()
        schedulerG.step()
        decay_ratio =  schedulerD.gamma ** (schedulerD.step_size // schedulerD.last_epoch)
        # update dataset
        if (epoch + 1) % args.update_interval == 0 and fix_dataset_flag == False: #
            if epoch > 300:
                pdb.set_trace()
            min_distance, fig = update_dataset(sub_traj_dataset=dataset, netQ=netQ, encoder=encoder, device=device, label_dtype=label_dtype, threshold=args.threshold)
            if args.wandb:
                wandb.log({"figure/distances_of_c":wandb.Image(fig)})
            if min_distance > .3:#TODO when to stop updating dataset? Judge by min_distance
                fix_dataset_flag = True
        # testing
        # dict of state and pred context
        if (epoch  )% 10 == 0:
            # pdb.set_trace()
            state_and_pred_context = test_for_context(Qnet=netQ,  dataset=dataset,  num_test_samples=10, dtype=dtype)

            # plot
            fig_dir = os.path.join(os.path.dirname(args.results_pkl_path), 'visualize')
            os.makedirs(fig_dir,exist_ok='True')
            
            wandb_log_stamp = now if args.wandb else False
            plot_trajectory_circle(traj_context_data=state_and_pred_context,
                            figsize=(6, 6),                         save_path=fig_dir,  wandb_log_stamp=wandb_log_stamp, eposode=epoch)

def test_for_context(Qnet:nn.Module, dataset:dynamic_sub_traj_dataset, dtype, num_test_samples=10):
    Qnet.eval()
    results = {'true_traj_state': [],'pred_context': []}
    # get sub_trajs for each traj
    j_subtraj = 0
    for i_traj in range(num_test_samples):
        pred_context = []
        true_traj = []
        subtraj = dataset.sample_batch_index(j_subtraj)
        while subtraj.traj_index == i_traj: # within the same traj
            x_feat = subtraj.data.unsqueeze(0).type(dtype)
            logits_q = F.softmax(Qnet(x_feat), dim=-1)[0] #(c) dim=1
            # pdb.set_trace()
            for _ in range(subtraj.data.shape[0]):
                pred_context.append(logits_q.data.cpu().numpy())
            if args.debug:
                print(subtraj.data)
            true_traj.append(subtraj.data)
            j_subtraj += 1
            subtraj = dataset.sample_batch_index(j_subtraj)
        print(true_traj)
        # pdb.set_trace()
        results['true_traj_state'].append(torch.cat(true_traj, dim=0))
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
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging ' \
                              'training status')
    parser.add_argument('--update_interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before updating dataset ')
    parser.add_argument('--expert_path', default='../IL/h5_trajs/circle_trajs/meta_42_traj_50_circles',
                        metavar='G',
                        help='path to the expert trajectory files')

    parser.add_argument('--state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--action_size', type=int, default=4,
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
    parser.add_argument('--lambda_post', type=float, default=3., help='param for poset')
    parser.add_argument('--lambda_dist', type=float, default=.001, help='param for dist')
    parser.add_argument('--lambda_gen', type=float, default=1., help='param for gen')
    parser.add_argument('--threshold', type=float, default=.5, help='threshold for updating dataset')
    parser.add_argument('--distance_type', type=str, default='l1', help='l1 or l2 loss for adjacent c')


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
    parser.add_argument('--results_pkl_path', default='./results/test_predict/circle/context_4/pred_result_cp_1000.pth')

    # training hyperparams
    
    # debug
    parser.add_argument('--debug', action='store_true', help='whether print out debugging message')
    parser.add_argument('--wandb', action='store_true', help='whether save on wandb')
    parser.add_argument('--teacher_force', action='store_true', help='whether use true state in RNN generator')
    args = parser.parse_args()
    now = time.strftime(f"RMSProp-Gen{args.lambda_gen}_Post{args.lambda_post}_Dist{args.lambda_dist}-"+"%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    
    if args.wandb:
        wandb.init(project="infoGAN_circle", entity="evieq01")
        wandb.run.name = f'{now}'
        wandb.config.update(args)

    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global_args = args
    main(args)
