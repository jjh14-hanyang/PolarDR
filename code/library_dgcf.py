# -*- coding: utf-8 -*
'''
This is a supporting library with the code of the model.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
# import cPickle
# import gpustat
from itertools import chain
from tqdm.std import tqdm, trange # tqdm_notebook, tnrange
import csv

from common_transition import CommonTransitionMixin

PATH = "./"


total_reinitialization_count = 0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE DGCF MODULE
class DGCF(nn.Module, CommonTransitionMixin):
    def __init__(self, args, num_features, num_users, num_items):
        super(DGCF,self).__init__()

        

        print ("*** Initializing the DGCF model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.adj = args.adj
        self.no_zero = args.no_zero
        self.no_first = args.no_first
        self.method = args.method
        self.length = args.sample_length

        print ("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        if self.adj:
            rnn_input_size_items = rnn_input_size_users = 2 * self.embedding_dim + 1 + num_features
        elif self.no_zero:
            rnn_input_size_items = rnn_input_size_users = self.embedding_dim + num_features
        elif self.no_first:
            rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1
        else:
            rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features


        print ("Initializing user and item RNNs")
        # DGCF pre update
        self.pre_item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.pre_user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)
        

        print ("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)

        
        self.interaction = nn.Linear(self.embedding_dim + self.embedding_dim + num_features, self.embedding_dim, bias=False) #pre
        rating_free_datasets = {
            "kuairand_pure",
        }

        self.use_rating_attribute = (args.dataset.lower() not in rating_free_datasets)

        post_attr_input_dim = (self.embedding_dim + 5 if self.use_rating_attribute else self.embedding_dim)

        # forward_post - user update
        self.user_positive_attribute = nn.Linear(post_attr_input_dim, self.embedding_dim, bias=False)
        self.user_negative_attribute = nn.Linear(post_attr_input_dim, self.embedding_dim, bias=False)

        # forward_post - item update
        self.item_positive_attribute = nn.Linear(post_attr_input_dim, self.embedding_dim, bias=False)
        self.item_negative_attribute = nn.Linear(post_attr_input_dim, self.embedding_dim, bias=False)
        

        print("Initializing re-scaling layers")
        self.LeakyReLU = nn.LeakyReLU()

        self.ReScale_Pre_Inter_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Pre_Inter_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)
        self.ReScale_Pre_Neigh_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Pre_Neigh_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)
        
        self.ReScale_Post_Inter_first = nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=True)
        self.ReScale_Post_Inter_factor = nn.Linear(self.embedding_dim, 1, bias=True)
        self.ReScale_Post_Neigh_first = nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=True)
        self.ReScale_Post_Neigh_factor = nn.Linear(self.embedding_dim, 1, bias=True)

        print ("*** DGCF initialization complete ***\n\n")

        if self.adj or self.no_zero or self.no_first:
            print("Initializing aggregate layers")
            if self.method == 'mean' or self.method == 'attention':
                self.weigh_item = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.weigh_user = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)
            elif self.method == 'gat':
                self.weigh_item = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.weigh_user = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.linear_three1 = nn.Linear(self.embedding_dim, 1, bias=False)
                self.linear_three2 = nn.Linear(self.embedding_dim, 1, bias=False)
            elif self.method == 'lstm':
                self.item_lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
                self.user_lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
                self.weigh_cen = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.weigh_adj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
    def forward_pre(self, user_embeddings, item_embeddings, timediffs=None,  features=None, adj_embeddings=None, select=None):
        if select == 'item_update':
            if self.adj:
                input1 = torch.cat([user_embeddings, timediffs, features, adj_embeddings], dim=1)
                item_embedding_output = self.pre_item_rnn(input1, item_embeddings)
            elif self.no_zero:
                input1 = torch.cat([features, adj_embeddings], dim=1)
                item_embedding_output = self.pre_item_rnn(input1, user_embeddings)
            elif self.no_first:
                input1 = torch.cat([timediffs, adj_embeddings], dim=1)
                item_embedding_output = self.pre_item_rnn(input1, item_embeddings)
            else:
                input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
                item_embedding_output = self.pre_item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            if self.adj:
                input2 = torch.cat([item_embeddings, timediffs, features, adj_embeddings], dim=1) 
                user_embedding_output = self.pre_user_rnn(input2, user_embeddings) 
            elif self.no_zero:
                input2 = torch.cat([features, adj_embeddings], dim=1)
                user_embedding_output = self.pre_user_rnn(input2, item_embeddings)
            elif self.no_first:
                input2 = torch.cat([timediffs, adj_embeddings], dim=1)
                user_embedding_output = self.pre_user_rnn(input2, user_embeddings)
            else:
                input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
                user_embedding_output = self.pre_user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)


        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            return user_projected_embedding


    def aggregate_attention(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_update':
            user_em = self.weigh_user(center_embedding)
            item_em = self.weigh_item(embeddings)
        else:
            user_em = self.weigh_item(center_embedding)
            item_em = self.weigh_user(embeddings)
        alpha = self.linear_three(torch.sigmoid(item_em + user_em.unsqueeze(1)))
        fin_em = torch.sum(alpha*embeddings*mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return fin_em

    def aggregate_gat(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_update':
            user_em = self.weigh_user(center_embedding)
            item_em = self.weigh_item(embeddings)
        else:
            user_em = self.weigh_item(center_embedding)
            item_em = self.weigh_user(embeddings)
        alpha = torch.nn.LeakyReLU()(self.linear_three1(item_em).squeeze(-1) + self.linear_three2(user_em))
        zero_vec = -9e15 * torch.ones_like(mask).float().cuda()
        attention = torch.softmax(torch.where(mask.cuda()>0, alpha, zero_vec), dim=1).unsqueeze(-1)
        fin_em = torch.sum(attention * embeddings * mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return fin_em

    def aggregate_lstm(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        if select == 'user_update':
            out, _ = self.user_lstm(embeddings)
        else:
            out, _ = self.item_lstm(embeddings)
        lstm_em = out[torch.arange(embeddings.shape[0]), length_mask-1, :]
        fin_em = self.weigh_cen(center_embedding) + self.weigh_adj(lstm_em)
        return fin_em

    def aggregate_mean(self, embeddings, length_mask, max_length, center_embedding, select=None):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_update':
            em = self.weigh_item(embeddings)
        else:
            em = self.weigh_user(embeddings)
        em_mean = torch.div(torch.sum(em.mul(mask.unsqueeze(2).float().cuda()), 1) + center_embedding, length_mask.unsqueeze(1).float().cuda()+1)
        return em_mean


    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out
    
    

    def re_scaling(self, embeddings, target=None):
        if embeddings is None:
            print('No embeddings to re-scale')
            return embeddings

        # Re-scaling Network
        if target == 'Pre_Inter':
            Inter_first = self.LeakyReLU(self.ReScale_Pre_Inter_first(embeddings))
            Inter_factor = self.LeakyReLU(self.ReScale_Pre_Inter_factor(Inter_first))
            return Inter_factor*embeddings#, Inter_factor
        if target == 'Pre_History':
            Neigh_first = self.LeakyReLU(self.ReScale_Pre_Neigh_first(embeddings))
            Neigh_factor = self.LeakyReLU(self.ReScale_Pre_Neigh_factor(Neigh_first))
            return Neigh_factor*embeddings#, Neigh_factor
        if target == 'Post_Inter':
            Inter_first = self.LeakyReLU(self.ReScale_Post_Inter_first(embeddings))
            Inter_factor = self.LeakyReLU(self.ReScale_Post_Inter_factor(Inter_first))
            return Inter_factor*embeddings#, Inter_factor
        if target == 'Post_History':
            Neigh_first = self.LeakyReLU(self.ReScale_Post_Neigh_first(embeddings))
            Neigh_factor = self.LeakyReLU(self.ReScale_Post_Neigh_factor(Neigh_first))
            return Neigh_factor*embeddings#, Neigh_factor
        

def adj_pad(adj_seq):
    adjs = []
    length = [len(seq) for seq in adj_seq]
    max_length = max(length)
    for seq in adj_seq:
        adjs.append(list(seq) + (max_length - len(seq))*[0])
    return adjs, length, max_length


def adj_sample(adj_seq, sam_l):
    adjs = []
    length = [len(seq[:sam_l]) for seq in adj_seq]
    max_length = max(length)
    for seq in adj_seq:
        adjs.append(seq[::-1][:sam_l] + (max_length - len(seq[:sam_l]))*[0])
    return adjs, length, max_length
