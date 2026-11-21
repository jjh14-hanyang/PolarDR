from __future__ import division
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
import math, random
import torch
from collections import defaultdict
import os
import logging
import numpy as np

from common_transition import CommonTransitionMixin

PATH = "./"
total_reinitialization_count = 0


class MLP(nn.Module):
    def __init__(self, mlp_num_layers, in_dim, out_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.mlp_num_layers = mlp_num_layers
        self.out_dim = out_dim

        if self.mlp_num_layers == 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(in_dim, out_dim))
            for layer in range(mlp_num_layers - 2):
                self.linears.append(nn.Linear(out_dim, out_dim))
            self.linears.append(nn.Linear(out_dim, out_dim))

    def forward(self, x):
        if self.linear_or_not:
            x = self.linear(x)
            return x
        else:
            h = x
            for i in range(self.mlp_num_layers-1):
                h = self.linears[i](h)
                h = F.relu(h)
            h=self.linears[-1](h)
            return h


class NeuralKF(nn.Module):
    def __init__(self, num_mlp_layers_kf, num_inst, emb_dim, num_features, device):
        super(NeuralKF, self).__init__()
        self.num_inst = num_inst
        self.emb_dim = emb_dim
        self.device = device
        self.no_feature = False
        if num_features == 0:
            self.no_feature = True

        if not self.no_feature:
            self.tran_mlp = MLP(num_mlp_layers_kf, emb_dim + num_features, emb_dim)
        self.pred_mlp = MLP(num_mlp_layers_kf, emb_dim, emb_dim)
        self.K_mlp_in = MLP(num_mlp_layers_kf, self.emb_dim + 1, self.emb_dim)
        self.K_gru = nn.GRUCell(self.emb_dim, self.emb_dim)
        self.h = torch.zeros((num_inst, self.emb_dim)).to(self.device)
        self.K_mlp_out = MLP(num_mlp_layers_kf, self.emb_dim, self.emb_dim)

    def forward(self, ids, emb_prio, emb, dual, feat):
        if not self.no_feature:
            emb = self.tran_mlp(torch.cat([emb, feat], dim=1))
        emb_post_pred = self.pred_mlp(emb)
        emb_res = emb_post_pred - emb_prio
        z_res = torch.ones((emb.shape[0], 1)).to(self.device) - torch.sum(torch.mul(dual, emb_prio), dim=1, keepdim=True)

        self.K_in = self.K_mlp_in(torch.cat([emb_res, z_res], dim=1))
        h_state = self.h[ids, :]
        h_state = self.K_gru(self.K_in, h_state)
        self.h[ids, :] = h_state
        self.K = self.K_mlp_out(h_state)

        emb_post = emb_prio + self.K * z_res
        emb_regu = torch.norm(emb_post_pred - emb_post)
        return emb_post, emb_regu


class NeuFilter(nn.Module, CommonTransitionMixin):
    def __init__(self, args, num_features, num_users, num_items, device):
        super(NeuFilter, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.device=device

        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        self.user_kf = NeuralKF(args.num_layer_kf, num_users, self.embedding_dim, num_features, device)
        self.item_kf = NeuralKF(args.num_layer_kf, num_items, self.embedding_dim, num_features, device)

        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.pe_div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-(math.log(10000.0) / (self.embedding_dim)))).to(device)

        self.user_project_layer = MLP(args.num_layer_kf, self.embedding_dim, self.embedding_dim)
        self.item_project_layer = MLP(args.num_layer_kf, self.embedding_dim, self.embedding_dim)


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

        self.LeakyReLU = nn.LeakyReLU()

        self.ReScale_Post_Inter_first = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=True)
        self.ReScale_Post_Inter_factor = nn.Linear(self.embedding_dim, 1, bias=True)
        self.ReScale_Post_Neigh_first = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=True)
        self.ReScale_Post_Neigh_factor = nn.Linear(self.embedding_dim, 1, bias=True)

        

    def forward_pre(self, user_embeddings, item_embeddings, user_prior=None, item_prior=None, users=None, items=None, user_timediffs=None, item_timediffs=None, features=None, select=None):
        if select == 'user_update':
            user_embedding_output, user_emb_regu = self.user_kf(users, user_prior, user_embeddings, item_embeddings, features)
            return F.normalize(user_embedding_output), user_emb_regu

        elif select == 'item_update':
            item_embedding_output, item_emb_regu = self.item_kf(items, item_prior, item_embeddings, user_embeddings, features)
            return F.normalize(item_embedding_output), item_emb_regu

        elif select == 'project':
            user_positional_encoding = self.get_positional_encoding(user_timediffs)
            item_positional_encoding = self.get_positional_encoding(item_timediffs)
            user_projected_embedding = self.user_project_layer(user_embeddings+user_positional_encoding)
            item_projected_embedding = self.item_project_layer(item_embeddings+item_positional_encoding)
            return user_projected_embedding, item_projected_embedding

    def get_positional_encoding(self, positions):
        positional_encoding = torch.zeros(positions.shape[0], self.embedding_dim).to(self.device)
        positional_encoding[:, 0::2] = torch.sin(positions * self.pe_div_term)
        positional_encoding[:, 1::2] = torch.cos(positions * self.pe_div_term)
        return positional_encoding

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
        if target == 'Post_Inter':
            Inter_first = self.LeakyReLU(self.ReScale_Post_Inter_first(embeddings))
            Inter_factor = self.LeakyReLU(self.ReScale_Post_Inter_factor(Inter_first))
            return Inter_factor*embeddings#, Inter_factor
        if target == 'Post_History':
            Neigh_first = self.LeakyReLU(self.ReScale_Post_Neigh_first(embeddings))
            Neigh_factor = self.LeakyReLU(self.ReScale_Post_Neigh_factor(Neigh_first))
            return Neigh_factor*embeddings#, Neigh_factor
