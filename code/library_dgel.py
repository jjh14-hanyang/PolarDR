# -*- coding: utf-8 -*
'''
This is implementation and support of the model DGEL
If you use our code or our paper, please cite our paper:
Dynamic Graph Evolution Learning for Recommendation
published at SIGIR 2023
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math
import random
import sys
from collections import defaultdict
import os
# import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv
import time

from common_transition import CommonTransitionMixin

PATH = "./"
total_reinitialization_count = 0


# BPR Loss
class BPR_loss(nn.Module):
    def __init__(self):
        super(BPR_loss, self).__init__()

    def forward(self, user_embedding, item_embeddings, item_neg_embeddings):
        pos = torch.sum(user_embedding*item_embeddings, 1)
        neg = torch.sum(user_embedding*item_neg_embeddings, 1)
        sig = torch.clamp(torch.sigmoid(pos - neg), 1e-6)
        loss = -torch.mean(torch.log(sig))

        return loss


# A normal-linear Layer
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class DGEL(nn.Module, CommonTransitionMixin):
    def __init__(self, args, num_features, num_users, num_items, final_embedding_dim):
        super(DGEL,self).__init__()
        

        print("***** Initializing the DGEL model *****")
        self.embedding_dim = args.embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        self.sample_length = args.sample_length
        self.num_features = num_features

        # initial embeddings
        self.initial_user_embedding = nn.Parameter(torch.Tensor(final_embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(final_embedding_dim))

        # history enhancement (simple RNN) DGEL pre update
        rnn_input_size_items = rnn_input_size_users = self.final_embedding_dim*3
        self.pre_item_rnn = nn.RNNCell(rnn_input_size_users, self.final_embedding_dim)
        self.pre_user_rnn = nn.RNNCell(rnn_input_size_items, self.final_embedding_dim)

        

        # future drifting and prediction layers
        self.embedding_layer = NormalLinear(1, self.final_embedding_dim)

        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.final_embedding_dim * 2, self.item_static_embedding_size + self.final_embedding_dim)

        # 1: interaction-level update
        self.interaction_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_user_under_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_user_under_feature = nn.Linear(self.num_features, self.embedding_dim, bias=False)

        self.interaction_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_item_under_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_item_under_feature = nn.Linear(self.num_features, self.embedding_dim, bias=False)

        self.user_interaction_time_interval = nn.Linear(1, self.embedding_dim, bias=False)
        self.item_interaction_time_interval = nn.Linear(1, self.embedding_dim, bias=False)

        # 2: Neighbor-level update
        self.weigh_item_adj = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.weigh_user_adj = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.weigh_item_itself = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.weigh_user_itself = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)

        # 3: Interaction-Neighbor (local learning) update
        self.base_intensity_for_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_user_under_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_user_under_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)

        self.base_intensity_for_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_item_under_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_item_under_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)

        # 4: Rescaling network
        self.ReScale_Inter_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Inter_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)

        self.ReScale_Neigh_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Neigh_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)

        self.ReScale_Inter_Neigh_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Inter_Neigh_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)

        self.interaction = nn.Linear(self.embedding_dim + self.embedding_dim + num_features, self.embedding_dim, bias=False)
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
        self.ReScale_Pre_Interv_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Pre_Inter_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)
        self.ReScale_Pre_Neigh_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Pre_Neigh_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)
        
        self.ReScale_Post_Inter_first = nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=True)
        self.ReScale_Post_Inter_factor = nn.Linear(self.embedding_dim, 1, bias=True)
        self.ReScale_Post_Neigh_first = nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=True)
        self.ReScale_Post_Neigh_factor = nn.Linear(self.embedding_dim, 1, bias=True)


        # BPR loss
        self.bpr_loss = BPR_loss().cuda()

        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()

        print("***** DGEL initialization complete ****\n")

    def forward_pre(self, center_embeddings, inter_embeddings, local_embeddings, timediffs=None,  features=None, adj_embeddings=None, select=None):

        if select != 'project':
            # re-scaling on sub-embeddings
            inter_embeddings = self.re_scaling(inter_embeddings, 'Inter')
            adj_embeddings = self.re_scaling(adj_embeddings, 'Neigh')
            local_embeddings = self.re_scaling(local_embeddings, 'Inter_Neigh')
            inputs = torch.cat([inter_embeddings, adj_embeddings, local_embeddings], dim=1)

        if select == 'item_update':
            item_embedding_output = self.pre_item_rnn(inputs, center_embeddings)
            return item_embedding_output

        elif select == 'user_update':
            user_embedding_output = self.pre_user_rnn(inputs, center_embeddings)
            return user_embedding_output

        elif select == 'project':
            user_projected_embedding = self.context_convert(center_embeddings, timediffs, features)
            return user_projected_embedding

    # Inherent interaction update
    def interaction_aggregate(self, center_embeddings, addition_embedding, features, timediffs, target=None):
        if target == 'user':
            output_embedding = self.interaction_user(center_embeddings) + self.interaction_user_under_item(addition_embedding) \
                                 + self.interaction_user_under_feature(features) + self.user_interaction_time_interval(timediffs)
        if target == 'item':
            output_embedding = self.interaction_item(center_embeddings) + self.interaction_item_under_user(addition_embedding) \
                                 + self.interaction_item_under_feature(features) + self.item_interaction_time_interval(timediffs)
        return self.LeakyReLU(output_embedding).cuda()

    # time-decay neighbor GCN
    def neighbor_aggregate(self, center_embedding, adj_embeddings, length_mask, max_length, history_timediffer, target=None):
        mask = torch.arange(max_length)[None, :].cuda() < length_mask[:, None].cuda()
        weight = torch.softmax(history_timediffer, dim=1).cuda()

        if target == 'user':
            adj = self.weigh_user_adj(adj_embeddings)
            itself = self.weigh_user_itself(center_embedding)
        else:
            adj = self.weigh_item_adj(adj_embeddings)
            itself = self.weigh_item_itself(center_embedding)

        final_aggregation = itself + torch.sum(weight.unsqueeze(2)*adj*mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return self.LeakyReLU(final_aggregation).cuda()

    # def neighbor_aggregate(self, center_embedding, adj_embeddings, length_mask, max_length, history_timediffer, target=None):
    #     mask = torch.arange(max_length)[None, :].cuda() < length_mask[:, None].cuda()
    #     weight = torch.softmax(history_timediffer, dim=1).cuda()

    #     if target == 'user':
    #         adj = self.weigh_user_adj(adj_embeddings)
    #         itself = self.weigh_user_itself(center_embedding)
    #     else:
    #         adj = self.weigh_item_adj(adj_embeddings)
    #         itself = self.weigh_item_itself(center_embedding)

    #     adj_part = torch.sum(weight.unsqueeze(2) * adj * mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
    #     final_aggregation = itself + adj_part
    #     out = self.LeakyReLU(final_aggregation)

    #     if not hasattr(self, "_debug_neighbor_count"):
    #         self._debug_neighbor_count = 0

    #     if target == 'item' and max_length >= self.sample_length:
    #         print("========== neighbor_aggregate item debug ==========")
    #         print("sample_length:", self.sample_length)
    #         print("max_length:", max_length)
    #         print("center_embedding norm:", center_embedding.norm().item())
    #         print("adj_embeddings norm:", adj_embeddings.norm().item())
    #         print("linear adj norm:", adj.norm().item())
    #         print("itself norm:", itself.norm().item())
    #         print("adj_part norm:", adj_part.norm().item())
    #         print("final before relu norm:", final_aggregation.norm().item())
    #         print("final after relu norm:", out.norm().item())
    #         print("weight min/max:", weight.min().item(), weight.max().item())

    #     return out.cuda()

    # pooling for local symbiotic learning
    def excitement_aggregate(self, embeddings, length_mask, max_length):
        mask = torch.arange(max_length)[None, :].cuda() < length_mask[:, None].cuda()
        mask_embeddings = embeddings*mask.view(mask.shape[0], -1, 1).float().cuda()
        final_aggregation = (torch.sum(mask_embeddings, dim=1).cuda()) / (length_mask.unsqueeze(1).cuda())

        return final_aggregation.cuda()

    # local symbiotic learning
    def local_aggregate(self, user_embeddings, item_embeddings, user_excitement, item_excitement):
        user_level3_base = self.base_intensity_for_user(user_embeddings - item_embeddings)
        user_level3_item_ex = self.transfer_ex_for_user_under_item(item_excitement)
        user_level3_user_ex = self.transfer_ex_for_user_under_user(user_excitement)
        user_level3_out = user_level3_base + user_level3_item_ex + user_level3_user_ex

        item_level3_base = self.base_intensity_for_item(item_embeddings - user_embeddings)
        item_level3_user_ex = self.transfer_ex_for_item_under_user(user_excitement)
        item_level3_item_ex = self.transfer_ex_for_item_under_item(item_excitement)
        item_level3_out = item_level3_base + item_level3_user_ex + item_level3_item_ex

        return self.LeakyReLU(user_level3_out).cuda(), self.LeakyReLU(item_level3_out).cuda()

    # re-scaling network
    def re_scaling(self, embeddings, target=None):
        if embeddings is None:
            print('No embeddings to re-scale')
            return embeddings

        # Re-scaling Network
        if target == 'Inter':
            Inter_first = self.LeakyReLU(self.ReScale_Inter_first(embeddings))
            Inter_factor = self.LeakyReLU(self.ReScale_Inter_factor(Inter_first))
            return Inter_factor*embeddings
        if target == 'Neigh':
            Neigh_first = self.LeakyReLU(self.ReScale_Neigh_first(embeddings))
            Neigh_factor = self.LeakyReLU(self.ReScale_Neigh_factor(Neigh_first))
            return Neigh_factor*embeddings
        if target == 'Inter_Neigh':
            Inter_Neigh_first = self.LeakyReLU(self.ReScale_Inter_Neigh_first(embeddings))
            Inter_Neigh_factor = self.LeakyReLU(self.ReScale_Inter_Neigh_factor(Inter_Neigh_first))
            return Inter_Neigh_factor*embeddings
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
        

    # future drift
    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings.cuda()

    # prediction
    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out.cuda()

    # negative sampling
    def sample_for_BPR(self, user, id_length, adj, bpr_avoid=None):
        sample_length = len(adj)
        sample = []
        for i in range(sample_length):
            if len(set(adj[i])) >= id_length:
                print('It can not have negative sample, change loss function pls')
                break
            while 1:
                s = random.randint(0, id_length)
                if s not in adj[i]:
                    sample.append(s)
                    break
                '''
                if s not in bpr_avoid[user[i]]:
                    sample.append(s)
                    break
                '''

        return sample

    def adj_sample(self, adj_seq, sam_l, target=None):
        temp = -1000 if target == 'timediffer' else 0
        adjs = []
        length = [len(seq[:sam_l]) for seq in adj_seq]
        max_length = max(length)
        for seq in adj_seq:
            adjs.append(seq[::-1][:sam_l] + (max_length - len(seq[:sam_l]))*[temp])
        return adjs, length, max_length
    
    