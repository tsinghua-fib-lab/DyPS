from operator import index
import random, os
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal, Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import distributions as pyd, float32
from algo.utils.valuenorm import ValueNorm
from algo.utils.layers import GraphAttentionLayer
from  algo.cvae import *
from copy import deepcopy
from collections import namedtuple
import math
import scipy.signal
from collections import deque
policy_num=3
def layer_init(layer, std=np.sqrt(2), bias_const=0.0, init=True):
    if init == False:
        return layer
    torch.nn.init.orthogonal_(layer.weight, std)
    if 'bias' in dir(layer):
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def huber_loss(e, d):
    a = (torch.abs(e) <= d).float()
    b = (torch.abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (torch.abs(e) - d / 2)


def mse_loss(e):
    return e ** 2 / 2


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr - (0.5e-4)) * (epoch / float(total_num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class order_embedding(nn.Module):
    def __init__(self, grid_dim, time_dim, embedding_dim, contin_dim, init=True):
        super(order_embedding, self).__init__()
        self.grid_dim = grid_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.contin_dim = int(contin_dim)
        gain = nn.init.calculate_gain(['tanh', 'relu'][0])

        self.grid_embedding = layer_init(nn.Embedding(grid_dim, embedding_dim), std=gain, bias_const=0, init=init)
        self.contin_embedding = layer_init(nn.Linear(self.contin_dim, embedding_dim), std=gain, bias_const=0, init=init)
        self.order_layer2 = layer_init(nn.Linear(3 * embedding_dim, 1 * embedding_dim), std=gain, bias_const=0,
                                       init=init)
        self.order_layer3 = layer_init(nn.Linear(1 * embedding_dim, 1 * embedding_dim), std=1, bias_const=0, init=init)
        self.tanh = nn.Tanh()

    def forward(self, order):
        '''
        grid= order[:,:,:2].long()
        contin=order[:,:,2:].float()
        '''
        grid = order[..., :2].long()
        contin = order[..., 2:].float()
        grid_emb = self.tanh(self.grid_embedding(grid))
        contin_emb = self.tanh(self.contin_embedding(contin))
        # order_emb=torch.cat([grid_emb[:,:,0,:],grid_emb[:,:,1,:],contin_emb],dim=-1)
        order_emb = torch.cat([grid_emb[..., 0, :], grid_emb[..., 1, :], contin_emb], dim=-1)
        order_emb = self.tanh(self.order_layer2(order_emb))
        order_emb = self.order_layer3(order_emb)
        return order_emb


class state_embedding(nn.Module):
    def __init__(self, grid_dim, time_dim, embedding_dim, contin_dim, init=True):
        super(state_embedding, self).__init__()
        self.grid_dim = grid_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.contin_dim = int(contin_dim)
        gain = nn.init.calculate_gain(['tanh', 'relu'][0])

        self.grid_embedding = layer_init(nn.Embedding(grid_dim, embedding_dim), std=gain, bias_const=0, init=init)
        self.time_embedding = layer_init(nn.Embedding(time_dim, embedding_dim), std=gain, bias_const=0, init=init)
        self.contin_embedding = layer_init(nn.Linear(self.contin_dim, embedding_dim), std=gain, bias_const=0, init=init)
        self.state_layer2 = layer_init(nn.Linear(3 * embedding_dim, 1 * embedding_dim), std=gain, bias_const=0,
                                       init=init)
        self.tanh = nn.Tanh()

    def forward(self, state):
        '''
        time=state[:,0].long()
        grid= state[:,1].long()
        contin=state[:,2:].float()
        '''
        time = state[..., 0].long()
        grid = state[..., 1].long()
        contin = state[..., 2:].float()
        time_emb = self.tanh(self.time_embedding(time))
        grid_emb = self.tanh(self.grid_embedding(grid))
        contin_emb = self.tanh(self.contin_embedding(contin))
        state_emb = torch.cat([time_emb, grid_emb, contin_emb], dim=-1)
        state_emb = self.tanh(self.state_layer2(state_emb))
        return state_emb


class RNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num, init=True):
        super(RNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.init = init

        self.rnn = nn.GRU(input_dim, output_dim, num_layers=layer_num)
        # self.norm = nn.LayerNorm(outputs_dim)
        if self.init:
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    def forward(self, input, hidden):
        '''
        input: (seq length, batch size, feature dim)
        hidden:(layer num,  batch size, hidden dim)
        output:(seq length, batch size, output dim)
        '''
        output, hidden = self.rnn(input.unsqueeze(0), hidden)
        return output.squeeze(0), hidden


class GATLayer(nn.Module):
    def __init__(self, nfeat, nhid, output_dim, dropout, alpha, nheads, use_dropout=False, init=True):
        """Dense version of GAT."""
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.use_dropout = use_dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, use_dropout=use_dropout) for _
            in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, output_dim, dropout=dropout, alpha=alpha, concat=False,
                                           use_dropout=use_dropout)
        gain = nn.init.calculate_gain(['tanh', 'relu'][0])
        self.tanh = nn.Tanh()
        self.fc = layer_init(nn.Linear(output_dim, output_dim), std=gain, bias_const=0, init=init)

    def forward(self, x, adj):
        if self.use_dropout:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        if self.use_dropout:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return self.tanh(self.fc(x))
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)


class NeighborLayer(nn.Module):
    def __init__(self, input_dim, output_dim, init=True):
        super(NeighborLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = init
        self.tanh = nn.Tanh()
        gain = nn.init.calculate_gain(['tanh', 'relu'][0])
        self.fc = layer_init(nn.Linear(input_dim, output_dim), std=gain, bias_const=0, init=init)

    def forward(self, x, adj):
        output = torch.matmul(adj, x) / torch.sum(adj, dim=-1, keepdim=True)
        output = self.tanh(self.fc(output))
        return output


'''
class state_embedding(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim ,output_dim, contin_dim, init=True):
        super(state_embedding,self).__init__()
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.contin_dim = int(contin_dim)
        gain = nn.init.calculate_gain(['tanh', 'relu'][0])

        self.grid_embedding = layer_init( nn.Embedding(grid_dim,embedding_dim), std=gain, bias_const=0, init=init)
        self.time_embedding = layer_init( nn.Embedding(time_dim,embedding_dim), std=gain, bias_const=0, init=init )
        self.contin_embedding = layer_init( nn.Linear(self.contin_dim,embedding_dim), std=gain, bias_const=0, init=init)
        self.state_layer2 = layer_init( nn.Linear(3*embedding_dim,1*embedding_dim), std=gain, bias_const=0, init=init)
        self.state_layer3 = layer_init( nn.Linear(1*embedding_dim,output_dim), std=1, bias_const=0, init=init)
        self.tanh=nn.Tanh()

    def forward(self,state):
        time=state[:,0].long()
        grid= state[:,1].long()
        contin=state[:,2:].float()
        time_emb= self.tanh(self.time_embedding(time))
        grid_emb= self.tanh(self.grid_embedding(grid))
        contin_emb = self.tanh(self.contin_embedding(contin))
        state_emb=torch.cat([time_emb,grid_emb,contin_emb],dim=-1)
        state_emb= self.tanh(self.state_layer2(state_emb))
        state_emb = self.state_layer3(state_emb)
        return state_emb
'''
class actor_hrl(nn.Module):
    def __init__(
        self,
        input_size=40,
        hidden_size=32,
        output_size=2,
        embedding_dim=16,
        id_num=36
    ):
        super(actor_hrl, self).__init__()
        ### LSTM的参数情况
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.output_size=output_size
        self.embedding_dim=embedding_dim
        self.id_embedding = nn.Embedding(
            num_embeddings=id_num, embedding_dim=16)
        self.layer1 =nn.Linear(self.input_size,self.hidden_size)
        self.layer2 = nn.Linear(self.embedding_dim*2, 8)
        self.layer3 = nn.Linear(8, self.output_size)
        self.layer_cvae=nn.Linear(16,self.embedding_dim)
    def eva(self,z,u):
        z = F.elu(self.id_embedding(z))
        z = torch.matmul(z,u)
        action_prob = F.softmax(z, -1)
        return action_prob
    def forward(self,z,u):
        z=F.elu(self.id_embedding(z))
        z=torch.bmm(z,u)
        action_prob=F.softmax(z,-1)
        return action_prob




class critic_hrl(nn.Module):
    def __init__(
            self,
            input_size=40,
            hidden_size=32,
            output_size=6,
            embedding_dim=16,
            id_num=36
    ):
        super(critic_hrl, self).__init__()
        ### LSTM的参数情况
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.id_embedding = nn.Embedding(
            num_embeddings=id_num, embedding_dim=16)
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(policy_num, 1)
        self.layer3 = nn.Linear(8, 1)
        self.layer_cvae = nn.Linear(16, self.embedding_dim)
        self.layer4 = nn.Linear(id_num, 1)

    def forward(self, z, u):
        z = F.elu(self.id_embedding(z))
        z = F.elu(torch.bmm(z, u))
        z = F.elu(self.layer2(z))
        z=z.squeeze(-1)
        z = F.elu(self.layer4(z))

        return z

class Actor(nn.Module):
    def __init__(self, grid_dim, time_dim, embedding_dim, state_contin_dim, order_contin_dim, init=True, use_rnn=False,
                 use_GAT=False, use_neighbor_state=False, merge_method='cat', use_auxi=False, use_dropout=False):
        super(Actor, self).__init__()
        # order : [origin grid id, des grid id, pickup time ,duration, price ]
        self.grid_dim = grid_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.use_rnn = use_rnn
        self.use_GAT = use_GAT
        self.use_neighbor_state = use_neighbor_state
        self.use_dropout = use_dropout
        self.merge_method = merge_method
        self.use_auxi = use_auxi
        self.tanh = nn.Tanh()

        self.state_layer = state_embedding(grid_dim, time_dim, embedding_dim, state_contin_dim, init)
        self.order_layer = order_embedding(grid_dim, time_dim, embedding_dim, order_contin_dim, init)
        self.key_embedding_dim = embedding_dim

        self.key_layer = layer_init(nn.Linear(self.key_embedding_dim, embedding_dim), std=1, bias_const=0, init=init)
        gain = nn.init.calculate_gain(['tanh', 'relu'][0])
        if self.use_auxi:
            self.auxiliary_layer = layer_init(nn.Linear(self.embedding_dim, embedding_dim), std=gain, bias_const=0,
                                              init=init)

    def forward(self, state, order, mask, adj=None, hidden_state=None, scale=True, return_logit=False):
        # key embedding
        if mask.dtype is not torch.bool:
            mask = mask.bool()
        state_emb = self.state_layer(state)
        state_emb = self.key_layer(state_emb)
        order_emb = self.order_layer(order)
        compatibility = torch.squeeze(torch.matmul(state_emb[:, None, :], order_emb.transpose(-2, -1)), dim=1)
        if scale:
            compatibility /= math.sqrt(state_emb.size(-1))
        compatibility[~mask] = -math.inf
        if return_logit:
            return compatibility, hidden_state
        probs = F.softmax(compatibility, dim=-1)
        return probs, hidden_state

    def get_state_emb(self, state):
        state_emb = self.state_layer(state)
        return state_emb

    def auxiliary_emb(self, state):
        state_emb = self.state_layer(state)
        auxi_emb = self.tanh(self.auxiliary_layer(state_emb))
        return auxi_emb

    def multi_mask_forward(self, state, order, mask, adj=None, hidden_state=None, scale=True):
        # key embedding
        if mask.dtype is not torch.bool:
            mask = mask.bool()
        state_emb = self.state_layer(state)
        state_emb = self.key_layer(state_emb)
        order_emb = self.order_layer(order)
        # compatibility = torch.matmul(state_emb[:,None,:], order_emb.transpose(-2, -1))
        compatibility = torch.matmul(state_emb[..., None, :], order_emb.transpose(-2, -1))
        if scale:
            compatibility /= math.sqrt(state_emb.size(-1))
        # compatibility= compatibility.repeat(1,mask.shape[1],1)
        repeat_shape = [1 for _ in compatibility.shape]
        repeat_shape[-2] = mask.shape[-2]
        compatibility = compatibility.repeat(tuple(repeat_shape))
        compatibility[~mask] = -math.inf
        probs = F.softmax(compatibility, dim=-1)
        return probs, hidden_state

    def _distribution(self, state, order, mask):
        probs = self.forward(state, order, mask)
        return Categorical(probs=probs)


class Critic(nn.Module):
    def __init__(self, grid_dim, time_dim, embedding_dim, state_contin_dim, order_contin_dim, init=True, use_rnn=False,
                 use_GAT=False, use_neighbor_state=False, merge_method='cat', use_auxi=False):
        super(Critic, self).__init__()
        # order : [origin grid id, des grid id, pickup time ,duration, price ]
        self.grid_dim = grid_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.use_rnn = use_rnn
        self.use_GAT = use_GAT
        self.use_neighbor_state = use_neighbor_state
        self.merge_method = merge_method
        self.use_auxi = use_auxi
        self.tanh = nn.Tanh()

        self.state_layer = state_embedding(grid_dim, time_dim, embedding_dim, state_contin_dim, init)
        # self.order_layer = order_embedding(grid_dim,time_dim,embedding_dim, order_contin_dim)
        self.value_embedding_dim = embedding_dim
        self.value_layer = layer_init(nn.Linear(self.value_embedding_dim, 1), std=1, bias_const=0, init=init)
        gain = nn.init.calculate_gain(['tanh', 'relu'][0])
        if self.use_auxi:
            self.auxiliary_layer = layer_init(nn.Linear(self.embedding_dim, embedding_dim), std=gain, bias_const=0,
                                              init=init)

    def forward(self, state, adj=None, hidden_state=None):
        state_emb = self.state_layer(state)
        value = self.value_layer(state_emb)
        return value, hidden_state

    def get_state_emb(self, state):
        state_emb = self.state_layer(state)
        return state_emb

    def auxiliary_emb(self, state):
        state_emb = self.state_layer(state)
        auxi_emb = self.tanh(self.auxiliary_layer(state_emb))
        return auxi_emb


class MdpAgent(object):
    def __init__(self, time_len, node_num, gamma=0.99):
        self.gamma = gamma  # discount for future value
        self.time_len = time_len
        self.node_num = node_num
        self.value_state = np.zeros([time_len + 1, node_num])
        self.n_state = np.zeros([time_len + 1, node_num])
        self.cur_time = 0
        self.value_iter = []

    def get_value(self, order):
        # [begin node, end node, price, duration ,service type]
        value = order[2] + pow(self.gamma, order[3]) * self.value_state[
            min(self.cur_time + order[3], self.time_len), order[1]] - self.value_state[self.cur_time, order[0]]
        # value= pow(self.gamma,order[3])*self.value_state[min(self.cur_time+order[3],self.time_len), order[1]]  -self.value_state[self.cur_time, order[0]]
        return value

    def update_value(self, order, selected_ids, env):
        value_record = []
        for _node_id in env.get_node_ids():
            num = min(env.nodes[_node_id].idle_driver_num, len(selected_ids[_node_id]))
            for k in range(num):
                id = selected_ids[_node_id][k]
                o = order[_node_id][id]
                # self.n_state[self.cur_time, o[0]] += 1
                value = self.get_value(o)
                td = value
                # self.value_state[self.cur_time,o[0]]+= 1/self.n_state[self.cur_time, o[0]]*td
                self.value_state[self.cur_time, o[0]] = 199 / 200 * self.value_state[self.cur_time, o[0]] + 1 / 200 * td
                value_record.append(value)
        self.value_iter.append(np.mean(value_record))

    def save_param(self, dir):
        save_dict = {
            'value': self.value_state,
            'num': self.n_state
        }
        with open(dir + '/' + 'MDP.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

    def load_param(self, dir):
        with open(dir, 'rb') as f:
            MDP_param = pickle.load(f)
        self.value_state = MDP_param['value']
        self.n_state = MDP_param['num']


class PPO:
    """ build value network
    """

    def __init__(self,
                 env,
                 args,
                 device):
        self.set_seed(0)
        # param for env
        self.agent_num = args.grid_num
        self.TIME_LEN = args.TIME_LEN
        self.order_value = args.order_value
        self.order_grid = args.order_grid
        self.new_order_entropy = args.new_order_entropy
        self.remove_fake_order = args.remove_fake_order

        self.state_dim = env.get_state_space_node() + 1
        self.order_dim = 6

        if self.order_value:
            self.order_dim += 1
        if self.new_order_entropy:
            self.order_dim += 2

        self.max_order_num = 50
        self.action_dim = self.max_order_num

        self.hidden_dim = 128

        # param for hyperparameter
        self.total_steps = args.MAX_ITER
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.train_actor_iters = args.train_actor_iters
        self.train_critic_iters = args.train_critic_iters
        self.batch_size = int(args.batch_size)
        self.gamma = args.gamma
        self.lam = args.lam
        self.max_grad_norm = args.max_grad_norm
        self.clip_ratio = args.clip_ratio
        self.ent_factor = args.ent_factor
        self.adv_normal = args.adv_normal
        self.clip = args.clip
        self.grad_multi = args.grad_multi  # sum or mean
        self.minibatch_num = args.minibatch_num
        self.parallel_episode = args.parallel_episode
        self.parallel_way = args.parallel_way
        self.parallel_queue = args.parallel_queue

        self.use_orthogonal = args.use_orthogonal
        self.use_value_clip = args.use_value_clip
        self.use_valuenorm = args.use_valuenorm
        self.use_huberloss = args.use_huberloss
        self.huber_delta = 10
        self.use_lr_anneal = args.use_lr_anneal
        self.use_GAEreturn = args.use_GAEreturn
        self.use_rnn = args.use_rnn
        self.use_GAT = args.use_GAT
        self.use_dropout = args.use_dropout
        self.use_neighbor_state = args.use_neighbor_state
        self.adj_rank = args.adj_rank
        self.merge_method = args.merge_method
        self.use_auxi = args.use_auxi
        self.auxi_effi = args.auxi_effi
        self.use_regularize = args.use_regularize
        self.regularize_alpha = args.regularize_alpha
        self.use_fake_auxi = args.use_fake_auxi

        self.actor_decen = not args.actor_centralize
        self.critic_decen = not args.critic_centralize

        if self.use_GAT:
            self.adj_rank = 1
        self.clip_param = 0.2
        self.en_para = 0.01
        self.max_grad_norm = 0.5
        self.device = device
        if self.use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None
        # 定义上层结构
        self.hrl_actor = actor_hrl(output_size=policy_num).to(device)
        self.hrl_critic = critic_hrl().to(device)
        self.hrl_actor_optimizer = torch.optim.Adam(self.hrl_actor.parameters(), lr=self.actor_lr/10.0)
        self.hrl_critic_optimizer = torch.optim.Adam(self.hrl_critic.parameters(), lr=self.critic_lr/10.0)
        self.hrl_buffer = deque(maxlen=20)


        # 定义下层结构
        # 定义一堆list
        self.actor_l = []
        self.critic_l = []
        self.actor_optimizer_l = []
        self.critic_optimizer_l = []



        for inter in range(policy_num):
            self.actor_l.append(Actor(self.agent_num, self.TIME_LEN, 128, self.state_dim - 2, self.order_dim - 2,
                           self.use_orthogonal, self.use_rnn, self.use_GAT and (not self.actor_decen),
                           self.use_neighbor_state and (not self.actor_decen), self.merge_method,
                           self.use_auxi or self.use_fake_auxi > 0).to(self.device))
            self.critic_l.append(Critic(self.agent_num, self.TIME_LEN, 128, self.state_dim - 2, self.order_dim - 2,
                             self.use_orthogonal, self.use_rnn, self.use_GAT and (not self.critic_decen),
                             self.use_neighbor_state and (not self.critic_decen), self.merge_method,
                             self.use_auxi or self.use_fake_auxi > 0).to(self.device))
        for inter in range(policy_num):
            self.actor_optimizer_l.append(torch.optim.Adam(self.actor_l[inter].parameters(), lr=self.actor_lr))
            self.critic_optimizer_l.append(torch.optim.Adam(self.critic_l[inter].parameters(), lr=self.critic_lr))

        #### cvae
        self.cvae_l = CVAE(16, policy_num, 15,50).to(device)
        self.cvae_l_buffer = deque(maxlen=500000)
        self.cvae_l_optimizer = torch.optim.Adam(self.cvae_l.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss(reduction="mean")






        ### 全局critic
        # optimizers
        # self.actor = Actor(self.agent_num, self.TIME_LEN, 128, self.state_dim - 2, self.order_dim - 2,
        #                    self.use_orthogonal, self.use_rnn, self.use_GAT and (not self.actor_decen),
        #                    self.use_neighbor_state and (not self.actor_decen), self.merge_method,
        #                    self.use_auxi or self.use_fake_auxi > 0)
        self.critic = Critic(self.agent_num, self.TIME_LEN, 128, self.state_dim - 2, self.order_dim - 2,
                             self.use_orthogonal, self.use_rnn, self.use_GAT and (not self.critic_decen),
                             self.use_neighbor_state and (not self.critic_decen), self.merge_method,
                             self.use_auxi or self.use_fake_auxi > 0).to(self.device)

        # Set up optimizers for policy and value function
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)



        self.buffer = Replay_buffer(args.memory_size, self.state_dim, self.order_dim, self.action_dim, self.hidden_dim,
                                    self.max_order_num, self.agent_num, self.gamma, self.lam, self.adv_normal,
                                    parallel_queue=self.parallel_queue, value_normalizer=self.value_normalizer,
                                    use_GAEreturn=self.use_GAEreturn, actor_decen=self.actor_decen,
                                    critic_decen=self.critic_decen)

        self.step = 0

        self.env = env

        self.adj = self.compute_neighbor_tensor()

        print('PPO init')
    def set_replay_buffer(self,agent_num):
        self.replay_buffer_l = []
        for inter in range(policy_num):
            self.replay_buffer_l.append(Replay_buffer(self.memory_size, self.state_dim, self.order_dim, self.action_dim, self.hidden_dim,
                                    self.max_order_num, agent_num[inter], self.gamma, self.lam, self.adv_normal,
                                    parallel_queue=self.parallel_queue, value_normalizer=self.value_normalizer,
                                    use_GAEreturn=self.use_GAEreturn, actor_decen=self.actor_decen,
                                    critic_decen=self.critic_decen))
    def set_seed(self, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def move_device(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def save_param(self, save_dir, save_name='param'):
        state = {
            'step': self.step,
            'actor net': self.actor.state_dict(),
            'critic net': self.critic.state_dict(),
            'actor optimizer': self.actor_optimizer.state_dict(),
            'critic optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(state, save_dir + '/' + save_name + '.pkl')

    def load_param(self, load_dir, resume=False):
        state = torch.load(load_dir)
        self.actor.load_state_dict(state['actor net'])
        self.critic.load_state_dict(state['critic net'])

    def check_grad(self, net):
        for name, parms in net.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                  ' -->grad_value:', parms.grad)

    def compute_neighbor_tensor(self):
        neighbors = [node.layers_neighbors_id for node in self.env.nodes]
        adj = torch.zeros((self.agent_num, self.agent_num), dtype=torch.float32)
        for i in range(self.agent_num):
            adj[i, i] = 1
            for k in range(self.adj_rank):
                adj[i, neighbors[i][k]] = 1
        '''
        if self.phi_method==0:
            neighbor_tensor=torch.zeros((self.coop_scope+1,self.agent_num,self.agent_num),dtype=torch.float)
            for i in range(self.agent_num):
                neighbor_tensor[0,i,i]=1
                for rank in range(self.coop_scope):
                    neighbor_tensor[rank+1,i,neighbors[i][rank]]=1
            neighbor_tensor=neighbor_tensor.to(self.device)
            neighbor_tensor/= torch.sum(neighbor_tensor,dim=2,keepdim=True)
            neighbor_tensor=neighbor_tensor[:,:,:,None].transpose(0,3).squeeze(0)[:,:,None,:]
        elif self.phi_method==1:
            neighbor_tensor=torch.zeros((self.agent_num,self.agent_num),dtype=torch.float)
            for i in range(self.agent_num):
                neighbor_tensor[i,i]=1
                for rank in range(self.coop_scope):
                    neighbor_tensor[i,neighbors[i][rank]]=1
            neighbor_tensor=neighbor_tensor.to(self.device)
            neighbor_tensor/=torch.sum(neighbor_tensor,dim=1,keepdim=True)
        '''
        return adj.to(self.device)

    def process_state(self, s, t):
        s = np.stack(s, axis=0)
        feature_max = np.max(s[:, 1:], axis=0)
        feature_max[feature_max == 0] = 1
        s[:, 1:] /= feature_max
        '''
        onehot_grid_id = np.eye(self.agent_num)
        if self.state_time==0:
            state= np.concatenate([onehot_grid_id,s[:,1:]],axis=1)
        elif self.state_time==1:
            time = np.zeros((s.shape[0],1),dtype=np.float)
            time[:,0]= t/self.TIME_LEN
            state= np.concatenate([onehot_grid_id, time ,s[:,1:]],axis=1)
        elif self.state_time==2:
            time= np.zeros((s.shape[0],self.TIME_LEN))
            time[:, int(t)]=1
            state= np.concatenate([onehot_grid_id, time ,s[:,1:]],axis=1)
        '''
        time = np.zeros((s.shape[0], 1), dtype=np.float)
        time[:, 0] = t
        state = np.concatenate([time, s], axis=1)
        return torch.Tensor(state)

    def add_order_value(self, order_state):
        if self.order_value == False:
            return order_state
        for i in range(len(order_state)):
            for j, o in enumerate(order_state[i]):
                o += [self.MDP.get_value(o)]
        return order_state

    def remove_order_grid(self, order):
        if self.order_grid:
            return order
        else:
            order[:, :, :2] = 0
            return order

    def mask_fake(self, order, mask):
        if self.remove_fake_order == False:
            return mask
        else:
            return mask & (order[:, :, 4] < 0)

    def add_new_entropy(self, env, order):
        driver_num = torch.Tensor([node.idle_driver_num for node in env.nodes]) + 1e-5
        order_num = torch.Tensor([node.real_order_num for node in env.nodes]) + 1e-5
        driver_order = torch.stack([driver_num, order_num], dim=1)
        ORR_entropy = torch.min(driver_order, dim=1)[0] / torch.max(driver_order, dim=1)[0]
        node = order[:, :, :2].long()
        entropy_feature = ORR_entropy[node[:, :, 1]] - ORR_entropy[node[:, :, 0]]
        driver_num_feature = driver_num[node[:, :, 1]] - driver_num[node[:, :, 0]]
        order_num_feature = order_num[node[:, :, 1]] - order_num[node[:, :, 0]]
        order[:, :, 5] = entropy_feature
        order = torch.cat([order, driver_num_feature[:, :, None], order_num_feature[:, :, None]], -1)
        return order

    def process_order(self, order_state):
        # [begin node, end node, price, duration ,service type, entropy]
        order_num = [len(order_state[i]) for i in range(len(order_state))]
        assert np.max(order_num) <= self.max_order_num, 'order num overflow'
        order_dim_origin = self.order_dim - 2 if self.new_order_entropy else self.order_dim
        order = torch.zeros((self.agent_num, self.max_order_num, order_dim_origin), dtype=float32)
        mask = torch.zeros((self.agent_num, self.max_order_num), dtype=torch.bool)
        for i in range(len(order_state)):
            order[i, :order_num[i]] = torch.Tensor(order_state[i])
            mask[i, :order_num[i]] = 1
        if self.new_order_entropy:
            order = self.add_new_entropy(self.env, order)
        order[:, :, 2:] = torch.clamp(order[:, :, 2:], -10, 10)
        feature_scale = torch.max(torch.abs(order[:, :, 2:]))
        feature_scale[feature_scale == 0] = 1
        order[:, :, 2:] /= feature_scale

        '''
        price_scale= torch.max(order[:,:,2])
        dura_scale = torch.max(order[:,:,3])
        ent_scale = torch.max(order[:,:,5])
        price_scale = 1  if price_scale==0 else price_scale
        dura_scale = 1  if dura_scale==0 else dura_scale
        ent_scale = 1 if ent_scale==0 else ent_scale
        order[:,:,2]/=price_scale
        order[:,:,3]/=dura_scale
        order[:,:,5]/=ent_scale
        if self.order_value==True:
            value_scale = torch.max(order[:,:,6])
            value_scale = 1 if value_scale==0 else value_scale
            order[:,:,6]/=value_scale
        '''
        return order, mask

    def action_process(self, action):
        low = self.action_space_low
        high = self.action_space_high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action
    def action_hrl(self,id,z, device='cpu'):
        action_prob=self.hrl_actor.eva(torch.LongTensor(id).to(device),torch.FloatTensor(z).to(device))
        dis=Categorical(action_prob)
        action=dis.sample()
        action_prob=action_prob.cpu().data.numpy()
        action=action.cpu().data.numpy()
        return action_prob,action

    def action(self, state, order, state_rnn_actor, state_rnn_critic, mask, order_idx, device='cpu',
               random_action=False, sample=True, MDP=None, fleet_help=False, need_full_prob=False, random_fleet=False,cluster_dict_=None,shunxu_=None):
        """ Compute current action for all grids give states
        :param s: grid_num x stat_dim,
        :return:
        """
        if random_fleet:
            fake_num = torch.sum(order[:, :, 4] > 0, dim=-1)

        mask = mask.bool()

        state = state.to(device)
        action_logits = []
        value_output = []
        state_l = []
        with torch.no_grad():
            for inter in range(policy_num):
                index = cluster_dict_[inter]
                s_grid = state[index]
                mask_grid = mask[index].to(device)
                order_grid=order[index].to(device)
                action_logit_grid, state_rnn_actor = self.actor_l[inter](s_grid,order_grid, mask_grid,self.adj,state_rnn_actor.to(device), return_logit=True)
                action_logits.append(action_logit_grid.cpu())
                ### global
                # value_output_grid= 0.5*self.critic_l[inter](s_grid, self.adj, state_rnn_critic.to(device))[0]+0.5*self.critic(s_grid, self.adj, state_rnn_critic.to(device))[0]
                ### local
                value_output_grid,_ = self.critic_l[inter](s_grid, self.adj, state_rnn_critic.to(device))
                value_output.append(value_output_grid.cpu())
                state_l.append(s_grid.cpu())
        action_logits = torch.cat(action_logits, 0)
        action_logits = action_logits[shunxu_]
        value_output = torch.cat(value_output, 0)
        value_output = value_output[shunxu_]


        logits = action_logits
        value = value_output
        action = torch.zeros((self.agent_num, self.max_order_num), dtype=torch.float32)
        mask_order = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[1]), dtype=torch.bool)
        mask_action = torch.zeros((self.agent_num, self.max_order_num), dtype=torch.bool)
        driver_record = torch.zeros((self.agent_num,), dtype=torch.long)
        oldp = torch.zeros((self.agent_num, self.max_order_num), dtype=torch.float32)
        mask_agent = torch.ones((self.agent_num,), dtype=torch.bool)
        action_ids = []
        selected_idx = []
        # sample orders
        for i in range(state.shape[0]):
            max_driver_num = 50
            driver_num = self.env.nodes[i].idle_driver_num
            driver_num = min(driver_num, max_driver_num)
            driver_record[i] = driver_num
            if driver_num == 0 or len(order_idx[i]) == 1:
                choose = [0]
                mask_agent[i] = 0
            else:
                choose = []
                logit = logits[i][mask[i]]
                prob = F.softmax(logit,dim=-1)
                mask_d = mask[i]
                for d in range(driver_num):
                    mask_order[i, d] = mask_d.clone()
                    choose.append(torch.multinomial(prob, 1, replacement=True))
                    if random_fleet:
                        if order[i, choose[-1], 4] >= 0:
                            choose[-1] = torch.randint(0, fake_num[i] + 1, (1,))
                            # choose[-1]= torch.randint(fake_num[i],fake_num[i]+1,(1,))
                    mask_action[i, d] = 1
                    oldp[i, d] = prob[choose[-1]]
                    if order[i, choose[-1], 4] < 0:
                        mask_d[choose[-1]] = 0
                        logit[choose[-1]] = -math.inf
                        prob = F.softmax(logit,dim=-1)
                    if prob[0] == 1 and fleet_help == False:
                        break
            action[i, :len(choose)] = torch.Tensor(choose)
            action_ids.append([order_idx[i][idx] for idx in choose])
            selected_idx.append(choose)

        if need_full_prob:
            return action, value, oldp, mask_agent, mask_order, mask_action, action_ids, probs, driver_record
        return action, value, oldp, mask_agent, mask_order, mask_action, state_rnn_actor.cpu(), state_rnn_critic.cpu(), action_ids, selected_idx

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def compute_regularize(self, model):
        loss = 0
        for name, param in model.named_parameters():
            flag = False
            if 'weight' in name:
                if 'state' in self.use_regularize:
                    if 'state' in name:
                        flag = True
                else:
                    flag = True
            if flag:
                if 'L1' in self.use_regularize:
                    loss += torch.abs(param).sum()
                elif 'L2' in self.use_regularize:
                    loss += (param ** 2).sum() / 2
        return loss

    def add_L1_loss(self, model, l1_alpha):
        l1_loss = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_loss.append(torch.abs(param).sum())
        return l1_alpha * sum(l1_loss)

    def add_L2_loss(self, model, l1_alpha):
        l1_loss = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_loss.append((param ** 2).sum() / 2)
        return l1_alpha * sum(l1_loss)

    def split_batch(self, index, data, device='cpu'):
        batch = {}
        for key, value in data.items():
            batch[key] = value[index]
        return batch

    def update_cvae_l(self, device='cpu', writer=None):
        self.cvae_l.train()
        for inter in range(300):
            ex_batch = random.sample(self.cvae_l_buffer, 1280)
            id, state, action_prob = [], [], []
            for inter in ex_batch:
                id.append(inter[0][np.newaxis])
                state.append(inter[1][np.newaxis])
                action_prob.append(inter[2][np.newaxis])

            state = np.concatenate(state, 0)
            action_prob = np.concatenate(action_prob, 0)
            encoder_in = torch.LongTensor(np.concatenate(id, 0)).to(device)
            decoder_in = torch.FloatTensor(state).to(device)
            y = torch.FloatTensor(action_prob).to(device)
            self.cvae_l_optimizer.zero_grad()
            reconstruction, mu, logvar = self.cvae_l(encoder_in, decoder_in)
            bce_loss = self.criterion(reconstruction, y)
            loss = self.final_loss(bce_loss, mu, logvar)
            loss.backward()
            self.cvae_l_optimizer.step()
    def final_loss(self,bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        kl_weight = 0.0001
        BCE = bce_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + kl_weight * KLD
    def ppo_hrl_train(self,device):
        lenth = len(self.hrl_buffer)
        if lenth < 1:
            www = 1
            return www
        else:

            self.batch_size_h = lenth
            ex_batch = random.sample(self.hrl_buffer, self.batch_size_h)
            state, action, action_prob, reward,z= [], [], [], [],[]
            for inter in ex_batch:
                state.append(inter[0][np.newaxis])
                action.append(inter[1][np.newaxis])
                action_prob.append(inter[2][np.newaxis])
                reward.append(inter[3][np.newaxis])
                z.append(inter[4][np.newaxis])


            state = np.concatenate(state, 0)
            action = torch.FloatTensor(np.concatenate(action, 0)).to(device)
            action_prob = torch.FloatTensor(np.concatenate(action_prob, 0)).to(device)
            reward = torch.FloatTensor(np.concatenate(reward, 0)).to(device)
            z=torch.FloatTensor(np.concatenate(z, 0)).to(device)


            ### 更新critic
            with torch.no_grad():
                target_v = reward.reshape(-1,1)

            value_loss = F.smooth_l1_loss(
                self.hrl_critic(torch.LongTensor(state).to(device),z), target_v)
            self.hrl_critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.hrl_critic.parameters(), self.max_grad_norm)
            self.hrl_critic_optimizer.step()

            ### 更新actor
            advantage = (target_v - self.hrl_critic(torch.LongTensor(state).to(device),z)).detach()
            advantage = (advantage - torch.mean(advantage)) / torch.std(advantage)
            # 更新actor
            new_prob = self.hrl_actor(torch.LongTensor(state).to(device),z)
            action_prob_new = torch.sum(torch.sum(torch.mul(new_prob, action), -1),-1)
            old_action_prob = torch.sum(torch.sum(torch.mul(action_prob, action), -1),-1)
            ratio = (action_prob_new / old_action_prob).reshape(-1, 1)
            dist = Categorical(new_prob)
            en_prefer = dist.entropy()
            L1=ratio * advantage.reshape(-1,1)
            L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage.reshape(-1,1)
            # actor_loss = -torch.min(L1, L2).mean() - self.en_para * en_prefer.mean()  # MAX->MIN desent
            actor_loss = -torch.min(L1, L2).mean()
            self.hrl_actor_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy_prefer.parameters(), self.max_grad_norm)
            self.hrl_actor_optimizer.step()
            return en_prefer.mean()




    def update(self, device='cpu', writer=None):
        for inter in range(policy_num):

            data = self.replay_buffer_l[inter].get(device)

            record_entropy = []
            record_return = []
            record_actor_loss_origin = []
            record_critic_loss_origin = []
            record_actor_auxi_loss = []
            record_critic_auxi_loss = []

            # Train policy with multiple steps of gradient descent
            data_actor = {
                'state': data['state_actor'],
                'next_state': data['next_state_actor'],
                'order': data['order'],
                'action': data['action'],
                'advantage': data['advantage'],
                'oldp': data['oldp'],
                'mask_order': data['mask_order'],
                'mask_action': data['mask_action'],
                'mask_agent': data['mask_agent'],
                'state_rnn': data['state_rnn_actor']
            }
            data_size = data_actor['state'].shape[0]
            if self.parallel_way == 'mean':
                data_size = int(data_size / self.parallel_episode)
            if not self.actor_decen:
                batch_size = int(self.batch_size / self.agent_num)
            else:
                batch_size = self.batch_size
            batch_num = int(np.round(data_size / batch_size / self.minibatch_num))
            if batch_num==0:
                aaa=1
            else:
                for iter in range(self.train_actor_iters):
                    record_actor_loss = []
                    record_ratio_max = []
                    record_ratio_mean = []
                    record_KL = []
                    self.actor_optimizer_l[inter].zero_grad()
                    batch_num = int(np.round(data_size / batch_size / self.minibatch_num))
                    cnt = 0
                    thread = 1 if self.parallel_way == 'mix' else self.parallel_episode
                    for index in BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, True):
                        record_KL = []
                        for _ in range(thread):
                            # self.actor_optimizer.zero_grad()
                            loss_actor, actor_info = self.compute_loss_actor(self.split_batch(index, data_actor),inter)
                            kl = actor_info['kl']
                            loss_actor /= (batch_num * thread)
                            loss_actor.backward()
                            if iter == 0:
                                record_entropy.append(actor_info['entropy'])
                                record_actor_loss_origin.append(loss_actor.item())
                            record_KL.append(actor_info['kl'])
                            record_actor_loss.append(loss_actor.item())
                            record_ratio_max.append(actor_info['ratio_max'])
                            record_ratio_mean.append(actor_info['ratio_mean'])
                            record_actor_auxi_loss.append(actor_info['auxi_loss'])
                            # loss_iter+=loss_actor
                        if (cnt + 1) % batch_num == 0:
                            if np.mean(record_KL) < -0.01:
                                self.actor_optimizer_l[inter].zero_grad()
                                # continue
                            else:
                                nn.utils.clip_grad_norm_(self.actor_l[inter].parameters(), self.max_grad_norm)
                                self.actor_optimizer_l[inter].step()
                                self.actor_optimizer_l[inter].zero_grad()
                        cnt += 1
                    # if np.mean(record_KL)<-0.01:
                    # break

                # Value function learning
                data_critic = {
                    'state': data['state_critic'],
                    'next_state': data['next_state_critic'],
                    'ret': data['ret'],
                    'value': data['value'],
                    'state_rnn': data['state_rnn_critic']
                }
                data_size = data_critic['state'].shape[0]
                if self.parallel_way == 'mean':
                    data_size = int(data_size / self.parallel_episode)
                if not self.critic_decen:
                    batch_size = int(self.batch_size / self.agent_num)
                else:
                    batch_size = self.batch_size
                for iter in range(self.train_critic_iters):
                    record_critic_loss = []
                    self.critic_optimizer_l[inter].zero_grad()
                    batch_num = int(np.round(data_size / batch_size / self.minibatch_num))
                    cnt = 0
                    thread = 1 if self.parallel_way == 'mix' else self.parallel_episode
                    for index in BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, True):
                        for _ in range(thread):
                            loss_critic, critic_info = self.compute_loss_critic(self.split_batch(index, data_critic),inter)
                            loss_critic /= (batch_num * thread)
                            loss_critic.backward()
                            if iter == 0:
                                record_critic_loss_origin.append(loss_critic.item())
                                record_return.append(critic_info['ret'])
                            record_critic_loss.append(loss_critic.item())
                            record_critic_auxi_loss.append(critic_info['auxi_loss'])
                        # mpi_avg_grads(ac.v)    # average grads across MPI processes
                        if (cnt + 1) % batch_num == 0:
                            # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, norm_type=2)
                            nn.utils.clip_grad_norm_(self.critic_l[inter].parameters(), self.max_grad_norm)
                            self.critic_optimizer_l[inter].step()
                            self.critic_optimizer_l[inter].zero_grad()
                        cnt += 1

        # writer.add_scalar('train actor loss', np.mean(record_actor_loss_origin), global_step=self.step)
        # writer.add_scalar('train critic loss', np.mean(record_critic_loss_origin), global_step=self.step)
        # writer.add_scalar('train entropy', np.mean(record_entropy), global_step=self.step)
        # writer.add_scalar('train kl', np.mean(record_KL), global_step=self.step)
        # writer.add_scalar('train delta actor loss', np.mean(record_actor_loss) - np.mean(record_actor_loss_origin),
        #                   global_step=self.step)
        # writer.add_scalar('train delta critic loss', np.mean(record_critic_loss) - np.mean(record_critic_loss_origin),
        #                   global_step=self.step)
        # writer.add_scalar('train ratio max', np.mean(record_ratio_max), global_step=self.step)
        # writer.add_scalar('train ratio mean', np.mean(record_ratio_mean), global_step=self.step)
        # writer.add_scalar('train adv mean', data['advantage'].mean(), global_step=self.step)
        # writer.add_scalar('train adv std', data['advantage'].std(), global_step=self.step)
        # writer.add_scalar('train return', np.mean(record_return), global_step=self.step)
        # writer.add_scalar('train actor auxi loss', np.mean(record_actor_auxi_loss), global_step=self.step)
        # writer.add_scalar('train critic auxi loss', np.mean(record_critic_auxi_loss), global_step=self.step)
        # self.step += 1

    def update_center(self, device='cpu', writer=None):
        if self.use_lr_anneal:
            update_linear_schedule(self.critic_optimizer, self.step, self.total_steps, self.critic_lr)

        data = self.buffer.get(device)
        record_entropy = []
        record_return = []
        record_actor_loss_origin = []
        record_critic_loss_origin = []
        record_actor_auxi_loss = []
        record_critic_auxi_loss = []

        # Value function learning
        data_critic = {
            'state': data['state_critic'],
            'next_state': data['next_state_critic'],
            'ret': data['ret'],
            'value': data['value'],
            'state_rnn': data['state_rnn_critic']
        }
        data_size = data_critic['state'].shape[0]
        if self.parallel_way == 'mean':
            data_size = int(data_size / self.parallel_episode)
        if not self.critic_decen:
            batch_size = int(self.batch_size / self.agent_num)
        else:
            batch_size = self.batch_size
        for iter in range(self.train_critic_iters):
            record_critic_loss = []
            self.critic_optimizer.zero_grad()
            batch_num = int(np.round(data_size / batch_size / self.minibatch_num))
            cnt = 0
            thread = 1 if self.parallel_way == 'mix' else self.parallel_episode
            for index in BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, True):
                for _ in range(thread):
                    loss_critic, critic_info = self.compute_loss_critic_center(self.split_batch(index, data_critic))
                    loss_critic /= (batch_num * thread)
                    loss_critic.backward()
                    if iter == 0:
                        record_critic_loss_origin.append(loss_critic.item())
                        record_return.append(critic_info['ret'])
                    record_critic_loss.append(loss_critic.item())
                    record_critic_auxi_loss.append(critic_info['auxi_loss'])
                # mpi_avg_grads(ac.v)    # average grads across MPI processes
                if (cnt + 1) % batch_num == 0:
                    # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, norm_type=2)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()
                cnt += 1


    def compute_loss_actor(self, data,index):
        state, next_state, order, action, advantage, oldp, mask_order, mask_action, mask_agent, hidden_rnn_actor = data[
                                                                                                                       'state'], \
                                                                                                                   data[
                                                                                                                       'next_state'], \
                                                                                                                   data[
                                                                                                                       'order'], \
                                                                                                                   data[
                                                                                                                       'action'], \
                                                                                                                   data[
                                                                                                                       'advantage'], \
                                                                                                                   data[
                                                                                                                       'oldp'], \
                                                                                                                   data[
                                                                                                                       'mask_order'], \
                                                                                                                   data[
                                                                                                                       'mask_action'], \
                                                                                                                   data[
                                                                                                                       'mask_agent'], \
                                                                                                                   data[
                                                                                                                       'state_rnn']
        # Policy loss
        probs, _ = self.actor_l[index].multi_mask_forward(state, order, mask_order, self.adj, hidden_rnn_actor.unsqueeze(0))
        # newp = torch.gather(probs, 2, action[:,:,None]).squeeze(-1)
        newp = torch.gather(probs, -1, action[..., None]).squeeze(-1)
        ratio = newp / oldp
        ratio[~mask_action] = 0
        # new_prob[~action]=1
        # logp = torch.sum(torch.log(new_prob),dim=1,keepdim=True)
        # ratio = torch.exp(logp - logp_old)
        if self.grad_multi == 'sum':
            ratio_max = torch.max(torch.abs(ratio - 1)[mask_action]).item()
            ratio = torch.sum(ratio, dim=1, keepdim=True)
            if self.clip:
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                clip_adv = ratio * advantage
                loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()
            else:
                loss_pi = -(ratio * advantage)[mask_agent].mean()
        elif self.grad_multi == 'mean':
            ratio_max = torch.max(torch.abs(ratio - 1)[mask_action]).item()
            ratio_mean = torch.mean(torch.abs(ratio - 1)[mask_action]).item()
            # ratio=torch.sum(ratio,dim=1,keepdim=True)/torch.sum(mask_action,dim=1,keepdim=True)
            if self.clip:
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                # clip_adv=ratio*advantage
                loss_pi = -(torch.min(ratio * advantage, clip_adv))
                loss_pi[~mask_action] = 0
                loss_pi = (torch.sum(loss_pi, dim=-1) / torch.sum(mask_action, dim=-1))[mask_agent].mean()
                ratio = torch.sum(ratio, dim=-1, keepdim=True) / torch.sum(mask_action, dim=-1, keepdim=True)
            else:
                ratio = torch.sum(ratio, dim=1, keepdim=True) / torch.sum(mask_action, dim=1, keepdim=True)
                loss_pi = -(ratio * advantage)[mask_agent].mean()
        # ent= -torch.sum((probs[:,0]+1e-12)*torch.log(probs[:,0]+1e-12),dim=1)
        ent = -torch.sum((probs[..., 0, :] + 1e-12) * torch.log(probs[..., 0, :] + 1e-12), dim=-1)
        # ent[~mask_action[:,0]]=0
        ent = ent[mask_agent].mean()
        # ent=  -torch.sum((probs[:,:,0]+1e-12)*torch.log(probs[:,:,0]+1e-12),dim=1)[mask_agent].mean()
        loss_pi -= self.ent_factor * ent
        if self.use_auxi:
            auxi_loss = self.compute_loss_auxi(self.actor, state, next_state)
            loss_pi += self.auxi_effi * auxi_loss
        else:
            auxi_loss = np.array([0])
        if self.use_regularize is not 'None':
            loss_pi += self.regularize_alpha * self.compute_regularize(self.actor)
        if self.use_fake_auxi > 0:
            loss_pi += self.auxi_effi * self.compute_fake_loss(self.actor, state, next_state)

        # Useful extra info
        approx_kl = torch.log(ratio[mask_agent]).mean().item()
        # approx_kl=0
        entropy = ent.item()
        # ratio_max= torch.max(torch.abs(ratio.detach())).item()
        # entropy = pi.entropy().mean().item()
        # clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        # clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, auxi_loss=auxi_loss.item(), entropy=entropy, ratio_max=ratio_max,
                       ratio_mean=ratio_mean)
        return loss_pi, pi_info

    def compute_loss_auxi(self, net, state, next_state):
        with torch.no_grad():
            next_state_label = net.get_state_emb(next_state).detach()
        next_state_pred = net.auxiliary_emb(state)
        error_pred = next_state_pred - next_state_label
        if self.use_huberloss:
            auxi_loss = huber_loss(error_pred, self.huber_delta).mean()
        else:
            auxi_loss = mse_loss(error_pred).mean()
        return auxi_loss

    def compute_fake_loss(self, net, state, next_state):
        next_state_label = net.get_state_emb(next_state)
        next_state_pred = net.auxiliary_emb(state)
        e = next_state_pred
        d = next_state_label
        if self.use_fake_auxi == 1:
            fake_loss = e ** 2 / 2 + d * (torch.abs(e) - d / 2)
        elif self.use_fake_auxi == 2:
            e = e.detach()
            fake_loss = e ** 2 / 2 + d * (torch.abs(e) - d / 2)
        elif self.use_fake_auxi == 3:
            d = d.detach()
            fake_loss = e ** 2 / 2 + d * (torch.abs(e) - d / 2)
        elif self.use_fake_auxi == 4:
            a = (torch.abs(e) <= d).float()
            b = (torch.abs(e) > d).float()
            fake_loss = a * e ** 2 / 2 + b * d * (torch.abs(e) - d / 2)
        elif self.use_fake_auxi == 5:
            e = e.detach()
            a = (torch.abs(e) <= d).float()
            b = (torch.abs(e) > d).float()
            fake_loss = a * e ** 2 / 2 + b * d * (torch.abs(e) - d / 2)
        elif self.use_fake_auxi == 6:
            d = d.detach()
            a = (torch.abs(e) <= d).float()
            b = (torch.abs(e) > d).float()
            fake_loss = a * e ** 2 / 2 + b * d * (torch.abs(e) - d / 2)
        elif self.use_fake_auxi == 7:
            fake_loss = -d ** 2 / 2
        elif self.use_fake_auxi == 8:
            fake_loss = -torch.abs(d)
        return fake_loss.mean()

    def compute_loss_critic_center(self, data):
        state, next_state, ret, old_value, hidden_rnn_critic = data['state'], data['next_state'], data['ret'], data[
            'value'], data['state_rnn']
        critic_info = dict(ret=ret.mean().item())
        new_value, _ = self.critic(state, self.adj, hidden_rnn_critic.unsqueeze(0))
        if self.use_valuenorm:
            self.value_normalizer.update(ret)
            ret = self.value_normalizer.normalize(ret)
        if self.use_value_clip:
            value_pred_clipped = old_value + (new_value - old_value).clamp(-self.clip_ratio, self.clip_ratio)
            error_clipped = ret - value_pred_clipped
            error_original = ret - new_value
            if self.use_huberloss:
                value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
                value_loss_original = huber_loss(error_original, self.huber_delta)
            else:
                value_loss_clipped = mse_loss(error_clipped)
                value_loss_original = mse_loss(error_original)
            value_loss = torch.max(value_loss_original, value_loss_clipped).mean()
        else:
            error_original = ret - new_value
            if self.use_huberloss:
                value_loss = huber_loss(error_original, self.huber_delta).mean()
            else:
                value_loss = mse_loss(error_original).mean()
        if self.use_auxi:
            auxi_loss = self.compute_loss_auxi(self.critic, state, next_state)
            value_loss += self.auxi_effi * auxi_loss
        else:
            auxi_loss = np.array([0])
        if self.use_fake_auxi > 0:
            value_loss += self.auxi_effi * self.compute_fake_loss(self.critic, state, next_state)
        if self.use_regularize is not 'None':
            value_loss += self.regularize_alpha * self.compute_regularize(self.critic)
        critic_info['auxi_loss'] = auxi_loss.item()
        return value_loss, critic_info

    def compute_loss_critic(self, data,index):
        state, next_state, ret, old_value, hidden_rnn_critic = data['state'], data['next_state'], data['ret'], data[
            'value'], data['state_rnn']
        critic_info = dict(ret=ret.mean().item())
        new_value, _ = self.critic_l[index](state, self.adj, hidden_rnn_critic.unsqueeze(0))
        if self.use_valuenorm:
            self.value_normalizer.update(ret)
            ret = self.value_normalizer.normalize(ret)
        if self.use_value_clip:
            value_pred_clipped = old_value + (new_value - old_value).clamp(-self.clip_ratio, self.clip_ratio)
            error_clipped = ret - value_pred_clipped
            error_original = ret - new_value
            if self.use_huberloss:
                value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
                value_loss_original = huber_loss(error_original, self.huber_delta)
            else:
                value_loss_clipped = mse_loss(error_clipped)
                value_loss_original = mse_loss(error_original)
            value_loss = torch.max(value_loss_original, value_loss_clipped).mean()
        else:
            error_original = ret - new_value
            if self.use_huberloss:
                value_loss = huber_loss(error_original, self.huber_delta).mean()
            else:
                value_loss = mse_loss(error_original).mean()
        if self.use_auxi:
            auxi_loss = self.compute_loss_auxi(self.critic_l[index], state, next_state)
            value_loss += self.auxi_effi * auxi_loss
        else:
            auxi_loss = np.array([0])
        if self.use_fake_auxi > 0:
            value_loss += self.auxi_effi * self.compute_fake_loss(self.critic_l[index], state, next_state)
        if self.use_regularize is not 'None':
            value_loss += self.regularize_alpha * self.compute_regularize(self.critic_l[index])
        critic_info['auxi_loss'] = auxi_loss.item()
        return value_loss, critic_info


class Replay_buffer():
    def __init__(self, capacity, state_dim, order_dim, action_dim, hidden_dim, max_order_num, agent_num, gamma=0.99,
                 lam=0.95, adv_normal=True, parallel_queue=False, value_normalizer=None, use_GAEreturn=False,
                 actor_decen=True, critic_decen=True):
        self.capacity = capacity
        self.agent_num = agent_num
        self.order_dim = order_dim
        self.max_order_num = max_order_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        assert self.action_dim == self.max_order_num, 'action dim error'

        self.state_pool = torch.zeros((self.agent_num, self.capacity, state_dim)).float()
        self.next_state_pool = torch.zeros((self.agent_num, self.capacity, state_dim)).float()
        self.order_pool = torch.zeros((self.agent_num, self.capacity, max_order_num, order_dim)).float()
        self.state_rnn_actor_pool = torch.zeros((self.agent_num, self.capacity, hidden_dim)).float()
        self.state_rnn_critic_pool = torch.zeros((self.agent_num, self.capacity, hidden_dim)).float()
        self.action_pool = torch.zeros((self.agent_num, self.capacity, max_order_num)).long()
        self.reward_pool = torch.zeros((self.agent_num, self.capacity, 1)).float()
        self.advantage_pool = torch.zeros((self.agent_num, self.capacity, 1)).float()
        self.return_pool = torch.zeros((self.agent_num, self.capacity, 1)).float()
        self.value_pool = torch.zeros((self.agent_num, self.capacity, 1)).float()
        self.oldp_pool = torch.zeros((self.agent_num, self.capacity, max_order_num)).float()
        self.mask_order_pool = torch.zeros((self.agent_num, self.capacity, max_order_num, max_order_num),
                                           dtype=torch.bool)
        self.mask_action_pool = torch.zeros((self.agent_num, self.capacity, max_order_num), dtype=torch.bool)
        self.mask_agent_pool = torch.zeros((self.agent_num, self.capacity), dtype=torch.bool)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.adv_normal = adv_normal
        self.parallel_queue = parallel_queue
        self.value_normalizer = value_normalizer
        self.use_GAEreturn = use_GAEreturn
        self.actor_decen = actor_decen
        self.critic_decen = critic_decen

    def condition_reshape(self, tensor, shape, condition):
        if condition:
            return torch.reshape(tensor, shape)
        else:
            return tensor.transpose(0, 1)

    def push(self, state, next_state, order, action, reward, value, p, mask_order, mask_action, mask_agent,
             state_rnn_actor, state_rnn_critic):
        assert self.ptr < self.capacity
        self.state_pool[:, self.ptr] = state
        self.next_state_pool[:, self.ptr] = next_state
        self.order_pool[:, self.ptr] = order
        self.action_pool[:, self.ptr] = action
        self.reward_pool[:, self.ptr] = reward
        self.value_pool[:, self.ptr] = value
        self.oldp_pool[:, self.ptr] = p
        self.mask_order_pool[:, self.ptr] = mask_order
        self.mask_agent_pool[:, self.ptr] = mask_agent
        self.mask_action_pool[:, self.ptr] = mask_action
        self.state_rnn_actor_pool[:, self.ptr] = state_rnn_actor
        self.state_rnn_critic_pool[:, self.ptr] = state_rnn_critic
        self.ptr += 1

    def finish_path(self, last_val=0):
        # path_slice = slice(self.path_start_idx, self.ptr)
        reward = torch.cat([self.reward_pool[:, self.path_start_idx:self.ptr], last_val[:, None, :]], dim=1)
        value = torch.cat([self.value_pool[:, self.path_start_idx:self.ptr], last_val[:, None, :]], dim=1)
        if self.value_normalizer is not None:
            value = self.value_normalizer.denormalize(value)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = reward[:, :-1] + self.gamma * value[:, 1:] - value[:, :-1]
        advantage = torch.zeros(deltas.shape, dtype=torch.float32)
        advantage[:, -1] = deltas[:, -1]
        ret = torch.zeros(deltas.shape, dtype=torch.float32)
        ret[:, -1] = reward[:, -2]
        for i in range(deltas.shape[1] - 2, -1, -1):
            advantage[:, i] = deltas[:, i] + advantage[:, i + 1] * (self.gamma * self.lam)
            ret[:, i] = self.gamma * ret[:, i + 1] + reward[:, i]
        # self.adv_buf[:,self.path_start_idx:self.prt] = scipy.signal.lfilter(deltas, self.gamma * self.lam)
        self.advantage_pool[:, self.path_start_idx:self.ptr] = advantage
        if self.use_GAEreturn:
            self.return_pool[:, self.path_start_idx:self.ptr] = advantage + value[:, :-1]
        else:
            self.return_pool[:, self.path_start_idx:self.ptr] = ret
        self.path_start_idx = self.ptr

    def sample(self, batch_size):
        index = np.random.choice(range(min(self.capacity, self.num_transition)), batch_size, replace=False)
        bn_s, bn_a, bn_r, bn_s_, bn_seq_, bn_d, bn_d_seq = self.state_pool[index], self.action_pool[index], \
                                                           self.reward_pool[index], \
                                                           self.next_state_pool[index], self.next_seq_pool[index], \
                                                           self.done_pool[index], self.done_seq_pool[index]

        return bn_s, bn_a, bn_r, bn_s_, bn_seq_, bn_d, bn_d_seq

    def normalize(self, input, flag):
        if flag == False:
            return input
        else:
            mean = torch.mean(input)
            std = torch.sqrt(torch.mean((input - mean) ** 2))
            if std == 0:
                std = 1
            return (input - mean) / std

    def get(self, device='cpu', writer=None):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr <= self.capacity

        record_ptr = self.ptr
        if self.parallel_queue:
            self.ptr = self.capacity

        '''
        if self.adv_normal:
            adv_mean=torch.mean(self.advantage_pool[:,:self.ptr])
            adv_std= torch.sqrt(torch.mean((self.advantage_pool[:,:self.ptr]-adv_mean)**2))
            if adv_std==0:
                adv_std=1
            self.advantage_pool = (self.advantage_pool-adv_mean)/adv_std
            #self.advantage_pool = self.advantage_pool/adv_std
        '''
        '''
        data = dict(
            state= torch.reshape(self.state_pool[:,:self.ptr],(-1,self.state_dim)).to(device), 
            order= torch.reshape(self.order_pool[:,:self.ptr],(-1,self.max_order_num,self.order_dim)).to(device),
            action= torch.reshape(self.action_pool[:,:self.ptr],(-1,self.max_order_num)).to(device),
            ret= torch.reshape(self.return_pool[:,:self.ptr],(-1,1)).to(device),
            value= torch.reshape(self.value_pool[:,:self.ptr],(-1,1)).to(device),
            #advantage= torch.reshape(self.advantage_pool[:,:self.ptr],(-1,1)).to(device),
            advantage= torch.reshape(self.normalize(self.advantage_pool[:,:self.ptr],self.adv_normal),(-1,1)).to(device),
            oldp= torch.reshape(self.oldp_pool[:,:self.ptr],(-1,self.max_order_num)).to(device),
            mask_order= torch.reshape(self.mask_order_pool[:,:self.ptr],(-1,self.max_order_num, self.max_order_num)).to(device),
            mask_agent= torch.reshape(self.mask_agent_pool[:,:self.ptr],(-1,)).to(device),
            mask_action= torch.reshape(self.mask_action_pool[:,:self.ptr],(-1,self.max_order_num)).to(device),
            state_rnn_actor = torch.reshape(self.state_rnn_actor_pool[:,:self.ptr],(-1,self.hidden_dim)).to(device),
            state_rnn_critic = torch.reshape(self.state_rnn_critic_pool[:,:self.ptr],(-1,self.hidden_dim)).to(device)
            )
        '''
        data = dict(
            state_actor=self.condition_reshape(self.state_pool[:, :self.ptr], (-1, self.state_dim),
                                               self.actor_decen).to(device),
            next_state_actor=self.condition_reshape(self.next_state_pool[:, :self.ptr], (-1, self.state_dim),
                                                    self.actor_decen).to(device),
            state_critic=self.condition_reshape(self.state_pool[:, :self.ptr], (-1, self.state_dim),
                                                self.critic_decen).to(device),
            next_state_critic=self.condition_reshape(self.next_state_pool[:, :self.ptr], (-1, self.state_dim),
                                                     self.critic_decen).to(device),
            order=self.condition_reshape(self.order_pool[:, :self.ptr], (-1, self.max_order_num, self.order_dim),
                                         self.actor_decen).to(device),
            action=self.condition_reshape(self.action_pool[:, :self.ptr], (-1, self.max_order_num),
                                          self.actor_decen).to(device),
            ret=self.condition_reshape(self.return_pool[:, :self.ptr], (-1, 1), self.critic_decen).to(device),
            value=self.condition_reshape(self.value_pool[:, :self.ptr], (-1, 1), self.critic_decen).to(device),
            # advantage= torch.reshape(self.advantage_pool[:,:self.ptr],(-1,1)).to(device),
            advantage=self.condition_reshape(self.normalize(self.advantage_pool[:, :self.ptr], self.adv_normal),
                                             (-1, 1), self.actor_decen).to(device),
            oldp=self.condition_reshape(self.oldp_pool[:, :self.ptr], (-1, self.max_order_num), self.actor_decen).to(
                device),
            mask_order=self.condition_reshape(self.mask_order_pool[:, :self.ptr],
                                              (-1, self.max_order_num, self.max_order_num), self.actor_decen).to(
                device),
            mask_agent=self.condition_reshape(self.mask_agent_pool[:, :self.ptr], (-1,), self.actor_decen).to(device),
            mask_action=self.condition_reshape(self.mask_action_pool[:, :self.ptr], (-1, self.max_order_num),
                                               self.actor_decen).to(device),
            state_rnn_actor=self.condition_reshape(self.state_rnn_actor_pool[:, :self.ptr], (-1, self.hidden_dim),
                                                   self.actor_decen).to(device),
            state_rnn_critic=self.condition_reshape(self.state_rnn_critic_pool[:, :self.ptr], (-1, self.hidden_dim),
                                                    self.critic_decen).to(device)
        )
        # size=self.ptr*self.agent_num
        if self.parallel_queue:
            self.ptr = record_ptr
            if self.ptr == self.capacity:
                self.ptr = 0
                self.path_start_idx = 0
        else:
            self.ptr = 0
            self.path_start_idx = 0
        return data


if __name__ == "__main__":
    prob = torch.Tensor([1, 2, 3]) / 0
    prob[0] = 1
    prob = torch.softmax(prob, 0)
    a = torch.multinomial(prob, 1, replacement=True)
    print(a)
    '''

    原始数据是origin_dataset
    有效数据对应的idx存在数组 valid_idx

    np.random.shuffle : 将valid_idx 打乱
    打乱之后按照batch大小取就好了
    假设每次取出的是 index
    然后 valid_idx[index]就得到了 origin_dataset中要取的哪些行
    origin_dataset[valid_idx[index]] 就是只有有效数据的

    '''