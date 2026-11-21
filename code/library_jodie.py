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
# import gpustat
from itertools import chain
from tqdm.std import tqdm, trange # tqdm_notebook, tnrange
import csv
import json

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


# THE JODIE MODULE
class JODIE(nn.Module, CommonTransitionMixin):
    def __init__(self, args, num_features, num_users, num_items):
        super(JODIE,self).__init__()

        print("*** Initializing the JODIE model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features

        print("Initializing user and item RNNs")

        # JODIE pre update
        self.pre_item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.pre_user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        

        print("Initializing linear layers")
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



        print("*** JODIE initialization complete ***\n\n")
        
    def forward_pre(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
            item_embedding_output = self.pre_item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.pre_user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            return user_projected_embedding

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