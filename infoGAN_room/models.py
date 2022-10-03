import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence
"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
	def __init__(self, hidden_size=128, input_size=12, max_len=10, output_size=2):
		super().__init__()
		self.max_len = max_len
		self.decoder1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
		self.decoder2 = nn.LSTMCell(input_size=hidden_size, hidden_size=output_size)
		# self.affine = nn.Linear(in_features=hidden_size, out_features=output_size)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		# self.decoder = nn.LSTM(hidden_size=hidden_size, input_size=input_size, batch_first=True, num_layers=2) #N, L , H_in

	def forward(self, x_0, z_c, teacher_force=False):
		B = z_c.size()[0]
		# x_0:PackedSequence -> B, len, dim
		# unpack
		x_unpacked, lens_unpacked = pad_packed_sequence(x_0, batch_first=True)
		# get init state
		out = torch.cat((x_unpacked[:, 0, :], z_c), dim=1) #(B, dim)
		# out_seq = torch.zeros(size=(x_0.size()[0], self.max_len, self.output_size)) # (B, max_len, dim)
		out_seq = []
		# h_x1, c_x1 = torch.zeros(size=(B, self.hidden_size)), torch.zeros(size=(B, self.hidden_size))
		# h_x2, c_x2 = torch.zeros(size=(B, self.output_size)), torch.zeros(size=(B, self.output_size))#TODO 3
		h_x1, c_x1 = self.decoder1( out)
		h_x2, c_x2 = self.decoder2(h_x1)
		out = torch.softmax(h_x2, dim=-1) #TODO 3
		out_seq.append(out)
		for t in range(1, max(lens_unpacked).item()):
			if teacher_force: # use true input!
				x_i = torch.cat((x_unpacked[:, t, :], z_c), dim=1)
			else:
				x_i = torch.cat((out.detach(), z_c), dim=1)
			h_x1, c_x1 = self.decoder1(x_i, (h_x1, c_x1))#(B, hidden, )
			# feed h_x1 to decoder2: h_x1=(B, hidden, )
			h_x2, c_x2 = self.decoder2(h_x1, (h_x2, c_x2))#(B, hidden, )
			# out = h_x2
			# out_x, (h_x, c_x) = self.decoder(y_i.unsqueeze(1))

			# h_x2 is output, 
			# out_seq[:, t] = h_x2
			# out = self.affine(h_x2)
			out = torch.softmax(out, dim=-1) #TODO 3
			out_seq.append(out)
			# topv, topi = out.topk(1) # (B, dimS_one_hot)
			# pdb.set_trace()
		
		out = torch.stack(out_seq) # max_len, B, dim
		# pdb.set_trace()
		out = out.permute((1, 0, 2)) # B, max_len, dim
		# same shape as input
		return pack_padded_sequence(input=out, lengths=lens_unpacked, batch_first=True, enforce_sorted=False)

'''
input: (B, L, D+c)
ourput: (B, 1)
'''
class Discriminator(nn.Module):
    
	def __init__(self, hidden_size=128, input_size=12):
		super().__init__()

		self.encoder = nn.LSTM(hidden_size=hidden_size, input_size=input_size, batch_first=True, num_layers=2) #N, L , H_in
        # self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        # self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.output = nn.Linear(hidden_size, 1)


	def forward(self, x):
		# pack
		# in = x_c[:, step_idx][x_c[:, step_idx] != 0]
		# pdb.set_trace()
		# tensor = nn.utils.rnn.pack_padded_sequence(x_c, length, batch_first=True)        
		o_n, (h_n, c_n) = self.encoder(x) # h, c shape as (2, 1, hidden)
		if isinstance(o_n, PackedSequence):
			o_unpacked, lens_unpacked = pad_packed_sequence(o_n, batch_first=True)
		else:
			o_unpacked = o_n
			# pdb.set_trace()
			# print(o_unpacked.shape)
		return self.output(torch.mean(o_unpacked, dim=-2)) # (B, hidden)
		# return self.output(h_n[-1]) # 2 for 2 layers
		# TODO activation function?


class QHead(nn.Module):
	def __init__(self, hidden_size=128, input_size=12, output_size=10):
		super().__init__()

		self.encoder = nn.LSTM(hidden_size=hidden_size, input_size=input_size, batch_first=True, num_layers=2) #N, L , H_in
        # self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        # self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.output = nn.Linear(hidden_size, output_size)
		# self.output_var = nn.Linear(hidden_size, 1)


	def forward(self, x):
		o_n, (h_n, c_n) = self.encoder(x) # h, c shape as (2, B, hidden)
		# print(o_n.shape)
		if isinstance(o_n, PackedSequence):
			o_unpacked, lens_unpacked = pad_packed_sequence(o_n, batch_first=True)
		else:
			o_unpacked = o_n
			# pdb.set_trace()
			# print(o_unpacked.shape)
		return self.output(torch.mean(o_unpacked, dim=-2)) # (B, hidden)
		# TODO activation function?

	# def forward(self, x):
	# 	x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

	# 	disc_logits = self.conv_disc(x).squeeze()

	# 	# Not used during training for celeba dataset.
	# 	mu = self.conv_mu(x).squeeze()
	# 	var = torch.exp(self.conv_var(x).squeeze())

	# 	return disc_logits, mu, var
