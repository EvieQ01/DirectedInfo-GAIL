import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence
out_vocab = {'BOS':-1, 'EOS': -1} # end of sentence is a sequence of boundary points.
"""
Architecture based on InfoGAN paper.
"""
class Encoder(nn.Module):
    def __init__(self, embed_size, c_size, vocab_size=None,  **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        self.s_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size) if vocab_size else None
        self.c_embed = nn.Embedding(num_embeddings=c_size, embedding_dim=embed_size)

        # self.rnn = nn.GRU(2*embed_size + noise_size, num_hiddens, num_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs_x, inputs_c):
        # 输入形状是(批量大小, 时间步数 )#。将输出互换样本维和时间步维
        embedding_s = self.s_embed(inputs_x) #(B, len) -> (B, len, dim)
        embedding_c = self.c_embed(inputs_c)
        # embedding = torch.cat((embedding_s, embedding_c, inputs_z), dim=-1)
        # return self.rnn(embedding, state)# (B, seq_len, input_size)
        return embedding_s, embedding_c

class Decoder(nn.Module):
    def __init__(self, input_size, num_hiddens, num_layers,
                drop_prob=0):
        super(Decoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU的输入包含attention输出的c和实际输入, 所以尺寸是 num_hiddens+embed_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_hiddens, 
                          num_layers=num_layers, dropout=drop_prob, batch_first=True)
        # self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, c):
        """
        cur_input shape: (batch, feat)
        state shape: (num_layers, batch, num_hiddens)
        c shape: (batch, embed)
        """
        # 使用注意力机制计算背景向量
        # c = attention_forward(self.attention, enc_states, state[-1])
        # c = enc_state
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, embed_size+embed_size)
        input_and_c = torch.cat((cur_input, c), dim=1)  #(B, embed + feat)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(1), state) #(B, feat)  #(2, B, feat))
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        # output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, input, c):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        output, state = self.rnn(torch.cat((input, c), dim=1).unsqueeze(1))
        return state

class Generator(nn.Module):
	def __init__(self, embed_size=10, hidden_size=128, state_size=2):
		super().__init__()
		# self.encoder = Encoder(vocab_size=output_size, embed_size=embed_size, noise_size=noise_size, c_size=c_size, \
		# 				num_hiddens=hidden_size, num_layers=2, drop_prob=0.2)
		self.decoder = Decoder(input_size=embed_size + state_size, num_hiddens=hidden_size, num_layers=2, drop_prob=0.2)
		self.affine = nn.Linear(in_features=hidden_size, out_features=state_size)
		self.hidden_size = hidden_size
		# self.decoder = nn.LSTM(hidden_size=hidden_size, input_size=input_size, batch_first=True, num_layers=2) #N, L , H_in

	def forward(self, x, noise_z, context, teacher_force=False):
		B = context.size()[0]
		# x_0:PackedSequence -> B, len
		# noise: (B, embed)
		# context: (B, embed)
		# input_c =  context.unsqueeze(1).expand_as(x) # (B, 1en, embed)
		# 初始化解码器的隐藏状态
		dec_state = self.decoder.begin_state(noise_z, c=context)
		# 解码器在最初时间步的输入是BOS
		dec_input = x[:, 0] # input = (B)
		out_seq = []
		step = 0
		for x_step in x.permute(1, 0, 2): # x shape: (batch, seq_len), y shape (B)
			# calculate over each timestep
			# decoder((B), (B, embed), (B, embed))
			# decoder((B), (B, embed), (B, embed))
			dec_output_feat, dec_state = self.decoder(cur_input=dec_input, state=dec_state, c=context)
			dec_output = self.affine(dec_output_feat)
			# pdb.set_trace()
			if teacher_force:
				dec_input = x_step  # 使用强制教学
			else:
				dec_input = dec_output.squeeze(1) #TODO argmax, 
			out_seq.append(dec_output.squeeze(1)) # append [B, dim]
			# num_not_pad_tokens += mask.sum().item()
			# # EOS后面全是PAD. 下面一行保证一旦遇到EOS接下来的循环中mask就一直是0
			# mask = mask * (x_step != out_vocab['EOS']).float()
			step += 1
		return torch.stack(out_seq).permute(1, 0, 2) # return (L, B, embed), (B, Len, embed)
	# class Generator(nn.Module):
# 	def __init__(self, hidden_size=128, input_size=12, max_len=10, output_size=2):
# 		super().__init__()
# 		self.max_len = max_len
# 		self.decoder1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
# 		self.decoder2 = nn.LSTMCell(input_size=hidden_size, hidden_size=output_size)
# 		# self.affine = nn.Linear(in_features=hidden_size, out_features=output_size)
# 		self.input_size = input_size
# 		self.hidden_size = hidden_size
# 		self.output_size = output_size
# 		# self.decoder = nn.LSTM(hidden_size=hidden_size, input_size=input_size, batch_first=True, num_layers=2) #N, L , H_in

# 	def forward(self, x_0, z_c, teacher_force=False):
# 		B = z_c.size()[0]
# 		# x_0:PackedSequence -> B, len, dim
# 		# unpack
# 		x_unpacked, lens_unpacked = pad_packed_sequence(x_0, batch_first=True)
# 		# get init state
# 		out = torch.cat((x_unpacked[:, 0, :], z_c), dim=1) #(B, dim)
# 		# out_seq = torch.zeros(size=(x_0.size()[0], self.max_len, self.output_size)) # (B, max_len, dim)
# 		out_seq = []
# 		h_x1, c_x1 = torch.zeros(size=(B, self.hidden_size)), torch.zeros(size=(B, self.hidden_size))
# 		h_x2, c_x2 = torch.zeros(size=(B, self.output_size)), torch.zeros(size=(B, self.output_size))#TODO 3
# 		# h_x1, c_x1 = self.decoder1( out)
# 		# h_x2, c_x2 = self.decoder2(h_x1)
# 		# out = torch.softmax(h_x2, dim=-1) #TODO 3
# 		# out_seq.append(out)
# 		for t in range(1, max(lens_unpacked).item()):
# 			if teacher_force: # use true input!
# 				x_i = torch.cat((x_unpacked[:, t, :], z_c), dim=1)
# 			else:
# 				x_i = torch.cat((out.detach(), z_c), dim=1)
# 			h_x1, c_x1 = self.decoder1(x_i, (h_x1, c_x1))#(B, hidden, )
# 			# feed h_x1 to decoder2: h_x1=(B, hidden, )
# 			h_x2, c_x2 = self.decoder2(h_x1, (h_x2, c_x2))#(B, hidden, )
# 			out = h_x2
# 			# out_x, (h_x, c_x) = self.decoder(y_i.unsqueeze(1))

# 			# h_x2 is output, 
# 			# out_seq[:, t] = h_x2
# 			# out = self.affine(h_x2)
# 			out = torch.softmax(out, dim=-1) #TODO 3
# 			out_seq.append(out)
# 			# topv, topi = out.topk(1) # (B, dimS_one_hot)
# 			# pdb.set_trace()
		
# 		out = torch.stack(out_seq) # max_len, B, dim
# 		pdb.set_trace()
# 		out = out.permute((1, 0, 2)) # B, max_len, dim
# 		# same shape as input
# 		return pack_padded_sequence(input=out, lengths=lens_unpacked, batch_first=True, enforce_sorted=False)

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
		# return self.output(o_unpacked[:, -1, :]) # (B, hidden)


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
		# return self.output(o_unpacked[:, -1, :]) # (B, hidden) #TODO

	# def forward(self, x):
	# 	x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

	# 	disc_logits = self.conv_disc(x).squeeze()

	# 	# Not used during training for celeba dataset.
	# 	mu = self.conv_mu(x).squeeze()
	# 	var = torch.exp(self.conv_var(x).squeeze())

	# 	return disc_logits, mu, var
