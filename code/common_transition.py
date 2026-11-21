import torch
import torch.nn as nn


class CommonTransitionMixin:

    def forward_fusion(self, pre_embedding, pos_embedding, neg_embedding, select):
        output = pre_embedding + pos_embedding - neg_embedding

        return output

    
    def forward_post(self, src_pos, src_neg, dst_pos, dst_neg, sign, all_dst_pos, all_dst_neg, adj_list, select, timediffs, rating, current_times):

        # MODULE 1
        if sign.dim() > 1:
            sign = sign.squeeze(-1)


        pos_mask = (sign == 1).unsqueeze(1)  # [B, 1]

        if self.use_rating_attribute:
            pos_input = torch.cat([dst_pos, rating], dim=1)
            neg_input = torch.cat([dst_neg, rating], dim=1)
        else:
            pos_input = dst_pos
            neg_input = dst_neg

        if select == 'user_update':
            pos_candidate = self.user_positive_attribute(pos_input) 
            neg_candidate = self.user_negative_attribute(neg_input) 
        else:
            pos_candidate = self.item_positive_attribute(pos_input) 
            neg_candidate = self.item_negative_attribute(neg_input)

        
        updated_pos = torch.where(pos_mask, pos_candidate, src_pos) 
        updated_neg = torch.where(pos_mask, src_neg, neg_candidate)


        # MODULE 2
        hist_pos, hist_neg = self._aggregate_signed_history(all_dst_pos=all_dst_pos, all_dst_neg=all_dst_neg, adj_list=adj_list, current_times=current_times)

        inter_embeddings = self.re_scaling(torch.cat([updated_pos, updated_neg], dim=1), 'Post_Inter')
        historical_embeddings = self.re_scaling(torch.cat([hist_pos, hist_neg], dim=1), 'Post_History')
        outputs = inter_embeddings + historical_embeddings

        return outputs


    def _aggregate_signed_history(self, all_dst_pos, all_dst_neg, adj_list, current_times):
        batch_size = len(adj_list)
        embedding_dim = all_dst_pos.shape[1]
        device = all_dst_pos.device

        if current_times.dim() == 1:
            current_times = current_times.unsqueeze(1) 
        current_times = current_times.to(device).float()

        max_neighbors = max((len(nb) for nb in adj_list), default=1)

        if max_neighbors == 0:
            hist_pos = torch.zeros(batch_size, embedding_dim, device=device)
            hist_neg = torch.zeros(batch_size, embedding_dim, device=device)
            return hist_pos, hist_neg

        PAD_IDX = 0

        nb_ids  = torch.full((batch_size, max_neighbors), PAD_IDX, dtype=torch.long,  device=device)
        nb_sign = torch.zeros(batch_size, max_neighbors,           dtype=torch.float, device=device)
        nb_ts   = torch.zeros(batch_size, max_neighbors,           dtype=torch.float, device=device)
        nb_last_ts = torch.full((batch_size, max_neighbors), -1.0, dtype=torch.float, device=device)
        valid_mask = torch.zeros(batch_size, max_neighbors,        dtype=torch.bool,  device=device)

        for b, neighbors in enumerate(adj_list):
            if len(neighbors) == 0:
                continue
            nbs, eidxs, tss, signs, last_tss  = zip(*neighbors)
            L = len(nbs)
            nb_ids [b, :L] = torch.tensor(nbs,   dtype=torch.long, device=device)
            nb_sign[b, :L] = torch.tensor(signs, dtype=torch.float, device=device)
            nb_ts  [b, :L] = torch.tensor(tss,   dtype=torch.float, device=device)
            nb_last_ts[b, :L] = torch.tensor(last_tss, dtype=torch.float, device=device)
            valid_mask[b, :L] = True

        pos_mask = valid_mask & (nb_sign == 1.0)   # [B, K]  bool
        neg_mask = valid_mask & (nb_sign != 1.0)   # [B, K]  bool

        
        flat_ids = nb_ids.view(-1)                          # [B*K]
        flat_pos_emb = all_dst_pos[flat_ids]                # [B*K, D]
        flat_neg_emb = all_dst_neg[flat_ids]                # [B*K, D]

        neighbor_pos_emb = flat_pos_emb.view(batch_size, max_neighbors, embedding_dim)  # [B, K, D]
        neighbor_neg_emb = flat_neg_emb.view(batch_size, max_neighbors, embedding_dim)  # [B, K, D]


        delta_t = current_times - nb_ts
        
        delta_t = delta_t.clamp(min=0.0)

        # time-decay softmax weight
        pos_alpha = self._masked_time_decay_softmax(delta_t, pos_mask)
        neg_alpha = self._masked_time_decay_softmax(delta_t, neg_mask)

        # weighted sum
        hist_pos = (neighbor_pos_emb * pos_alpha.unsqueeze(-1)).sum(dim=1)
        hist_neg = (neighbor_neg_emb * neg_alpha.unsqueeze(-1)).sum(dim=1)

        return hist_pos, hist_neg
    
    def _masked_time_decay_softmax(self, delta_t, mask, eps=1e-8):
        """
        delta_t: [B, K]
        mask:    [B, K]

        alpha_k = exp(- delta_t_k / max(delta_t)) / sum exp(...)
        """

        masked_delta = delta_t.masked_fill(~mask, 0.0)

        max_delta = masked_delta.max(dim=1, keepdim=True).values
        max_delta = max_delta.clamp(min=eps)

        score = -masked_delta / max_delta  # [B, K]

        exp_score = torch.exp(score) * mask.float()
        denom = exp_score.sum(dim=1, keepdim=True).clamp(min=eps)

        alpha = exp_score / denom

        return alpha
