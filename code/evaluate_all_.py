from library_data import *
from library_models import *
from library_jodie import *
from library_dgcf import *
from library_dgel import *
from library_neufilter import *
import time

def eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence,  rating_sequence, sign_sequence,
                   user_timediffs_sequence, user_pos_timediffs_sequence, user_neg_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, item_previous_userid_sequence,
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static,
                   start_idx, end_idx, num_users, full_adj_list, user_previous_sign_sequence, item_previous_sign_sequence,\
                   user_pos_sum, user_pos_cnt, user_neg_sum, user_neg_cnt, item_pos_sum, item_pos_cnt, item_neg_sum, item_neg_cnt,
                   user_adj=None, item_adj=None, user_timestamp_for_adj=None, item_timestamp_for_adj=None,
                   timedifference_sequence_for_adj=None, 
                   user_pre_embeddings=None, user_pos_embeddings=None, user_neg_embeddings=None,
                   item_pre_embeddings=None, item_pos_embeddings=None, item_neg_embeddings=None):


    # PERFORMANCE METRICS
    ranks = []
    inference_time_per_epoch = 0  

    torch.cuda.reset_peak_memory_stats()
    tbatch_start_time = None
    loss = 0
    # FORWARD PASS
    print("*** Making interaction predictions by forward pass (no t-batching) ***")
    # epoch_start_time = time.time()
    num_items = np.shape(item_embeddings)[0]
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / args.span_num

    user_last_ts_global = np.full(num_users, -1.0, dtype=np.float32)
    item_last_ts_global = np.full(num_items, -1.0, dtype=np.float32)

    for p in range(start_idx):
        u = int(user_sequence_id[p])
        it = int(item_sequence_id[p])
        ts = timestamp_sequence[p]

        user_last_ts_global[u] = ts
        item_last_ts_global[it] = ts

    for j in range(start_idx, end_idx):

        userid = int(user_sequence_id[j])
        itemid = item_sequence_id[j]
        feature = feature_sequence[j] if len(feature_sequence) > 0 else []
        rating =  rating_sequence[j]
        sign = sign_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        user_pos_timediff = user_pos_timediffs_sequence[j]
        user_neg_timediff = user_neg_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        if args.baseline == 'dgel':
            timestamp_for_adj = timedifference_sequence_for_adj[j]
        
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]
        userid_previous = item_previous_userid_sequence[j]
        current_tbatches_user_previous_sign = user_previous_sign_sequence[j]
        current_tbatches_item_previous_sign = item_previous_sign_sequence[j]

        torch.cuda.synchronize()
        epoch_start_time = time.time()
        # LOAD USER AND ITEM EMBEDDING
        user_embedding_input = user_embeddings[torch.tensor([userid], dtype=torch.long, device='cuda')]
        user_embedding_static_input = user_embeddings_static[torch.tensor([userid], dtype=torch.long, device='cuda')]
        item_embedding_input = item_embeddings[torch.tensor([itemid], dtype=torch.long, device='cuda')]
        item_embedding_static_input = item_embeddings_static[torch.tensor([itemid], dtype=torch.long, device='cuda')]
        if args.baseline in ['jodie', 'dgcf', 'dgel', 'neufilter']:
            feature_tensor = torch.tensor(feature, dtype=torch.float32, device='cuda').unsqueeze(0)
        user_timediffs_tensor = torch.tensor([user_timediff], dtype=torch.float32, device='cuda').unsqueeze(0)
        user_pos_timediffs_tensor = torch.tensor(
            [[user_pos_timediff]],
            dtype=torch.float32,
            device='cuda',
        )

        user_neg_timediffs_tensor = torch.tensor(
            [[user_neg_timediff]],
            dtype=torch.float32,
            device='cuda',
        )
        item_timediffs_tensor = torch.tensor([item_timediff], dtype=torch.float32, device='cuda').unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.tensor([itemid_previous], dtype=torch.long, device='cuda')]
        user_embedding_previous = user_embeddings[torch.tensor([userid_previous], dtype=torch.long, device='cuda')]
        current_tbatches_user_previous_sign_tensor = torch.tensor([current_tbatches_user_previous_sign], dtype=torch.float32, device='cuda')
        current_tbatches_item_previous_sign_tensor = torch.tensor([current_tbatches_item_previous_sign], dtype=torch.float32, device='cuda')

        raw_user_snap = [
            (nb, eidx, ts, s)
            for nb, eidx, ts, s in full_adj_list[userid]
            if eidx <= j
        ][-args.num_path_u_sample:]

        raw_item_snap = [
            (nb, eidx, ts, s)
            for nb, eidx, ts, s in full_adj_list[itemid + num_users]
            if eidx <= j
        ][-args.num_path_i_sample:]

        user_snap = [
            (nb - num_users, eidx, ts, s, item_last_ts_global[nb - num_users])
            for nb, eidx, ts, s in raw_user_snap
        ]

        item_snap = [
            (nb, eidx, ts, s, user_last_ts_global[nb])
            for nb, eidx, ts, s in raw_item_snap
        ]

        if args.baseline == 'tracerec':
            # # PROJECT USER EMBEDDING
            user_projected_embedding = model.forward_project(user_embedding_input, user_timediffs_tensor)
        elif args.baseline == 'dgel':
            user_projected_embedding = model.forward_pre(user_embedding_input, None, None, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        elif args.baseline == 'neufilter':
            user_projected_embedding, item_projected_embedding = model.forward_pre( user_embedding_input, item_embedding_input, user_timediffs=user_timediffs_tensor, item_timediffs=item_timediffs_tensor, select='project')
        else:
            user_projected_embedding = model.forward_pre(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')

        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.tensor([itemid_previous], dtype=torch.long, device='cuda')], user_embedding_static_input], dim=1)
        
        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        if args.online_test:
            loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())
        
        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS 
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1) 
        
        # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        ranks.append(true_item_rank)
        
        user_pre_embedding = user_pre_embeddings
        user_pos_embedding = user_pos_embeddings
        user_neg_embedding = user_neg_embeddings

        item_pre_embedding = item_pre_embeddings
        item_pos_embedding = item_pos_embeddings
        item_neg_embedding = item_neg_embeddings
        
        user_pre_embedding_input = user_pre_embedding[torch.tensor([userid], dtype=torch.long, device='cuda')] 
        item_pre_embedding_input = item_pre_embedding[torch.tensor([itemid], dtype=torch.long, device='cuda')]

        user_pos_embedding_input = user_pos_embedding[torch.tensor([userid], dtype=torch.long, device='cuda')] 
        item_pos_embedding_input = item_pos_embedding[torch.tensor([itemid], dtype=torch.long, device='cuda')]

        user_neg_embedding_input = user_neg_embedding[torch.tensor([userid], dtype=torch.long, device='cuda')] 
        item_neg_embedding_input = item_neg_embedding[torch.tensor([itemid], dtype=torch.long, device='cuda')]

        if args.baseline == 'tracerec':
            temporal_path_user = model.grab_subgraph([userid], [timestamp], args.len_path, args.num_path_u)
            temporal_path_item = model.grab_subgraph([itemid + num_users], [timestamp], args.len_path, args.num_path_i)

            subgraph_src_ = model.subgraph_tree2walk(np.array([userid]), np.array([timestamp]), temporal_path_user)
            subgraph_tgt_ = model.subgraph_tree2walk(np.array([itemid + num_users]), np.array([timestamp]), temporal_path_item)

            u_neighbor_records, u_time_decay_weights, u_masks, u_time_feat = model.forward_msg_time_delta(subgraph_src_) 
            i_neighbor_records, i_time_decay_weights, i_masks, i_time_feat = model.forward_msg_time_delta(subgraph_tgt_) 

            u_final_embeddings, u_valid_mask = model.get_valid_embeddings_vectorized(u_neighbor_records, u_time_feat, u_masks, user_pre_embedding, item_pre_embedding, 'user')
            i_final_embeddings, i_valid_mask = model.get_valid_embeddings_vectorized(i_neighbor_records, i_time_feat, i_masks, item_pre_embedding, user_pre_embedding, 'item')



            u_walk_final_embeddings = model.aggregate_embeddigs(u_final_embeddings, u_valid_mask, u_time_decay_weights, \
                                                                aggregation_method=args.aggregation_method, project=args.project)
            i_walk_final_embeddings = model.aggregate_embeddigs(i_final_embeddings, i_valid_mask, i_time_decay_weights, \
                                                                aggregation_method=args.aggregation_method, project=args.project)

            # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
            # user embedding update (dst: item)
            user_pre_emb = model.forward_pre(item_pre_embedding_input, u_walk_final_embeddings, user_timediffs_tensor) 
            # item embedding update (dst: user)
            item_pre_emb = model.forward_pre(user_pre_embedding_input, i_walk_final_embeddings, item_timediffs_tensor) 
        
        elif args.baseline == 'jodie':
            user_pre_emb = model.forward_pre(user_pre_embedding_input, item_pre_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
            item_pre_emb = model.forward_pre(user_pre_embedding_input, item_pre_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')
        
        elif args.baseline == 'dgcf':
            user_pre_input = user_pre_embedding_input
            item_pre_input = item_pre_embedding_input

            if args.adj or args.no_zero or args.no_first:
                if len(user_adj[userid]) == 0:
                    user_adj_embedding = torch.zeros(1, model.embedding_dim, device='cuda')
                else:
                    if not args.sample_length:
                        user_adj_, user_length_mask, user_max_length = adj_pad([list(user_adj[userid])])
                    else:
                        user_adj_, user_length_mask, user_max_length = adj_sample([list(user_adj[userid])], args.sample_length)

                    user_adj_em = item_pre_embedding[torch.LongTensor(user_adj_).cuda(), :]

                    if model.method == 'gat':
                        user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask), user_max_length, user_pre_input, select='user_update')
                    elif model.method == 'mean':
                        user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask), user_max_length, user_pre_input, select='user_update')
                    elif model.method == 'attention':
                        user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask), user_max_length, user_pre_input, select='user_update')
                    elif model.method == 'lstm':
                        user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask), user_max_length, user_pre_input, select='user_update')

                if len(item_adj[itemid]) == 0:
                    item_adj_embedding = torch.zeros(1, model.embedding_dim, device='cuda')
                else:
                    if not args.sample_length:
                        item_adj_, item_length_mask, item_max_length = adj_pad([list(item_adj[itemid])])
                    else:
                        item_adj_, item_length_mask, item_max_length = adj_sample([list(item_adj[itemid])], args.sample_length)

                    item_adj_em = user_pre_embedding[torch.LongTensor(item_adj_).cuda(), :]

                    if model.method == 'gat':
                        item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask), item_max_length, item_pre_input, select='item_update')
                    elif model.method == 'mean':
                        item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask), item_max_length, item_pre_input, select='item_update')
                    elif model.method == 'attention':
                        item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask), item_max_length, item_pre_input, select='item_update')
                    elif model.method == 'lstm':
                        item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask), item_max_length, item_pre_input, select='item_update')
            else:
                user_adj_embedding = None
                item_adj_embedding = None

            user_pre_emb = model.forward_pre(user_pre_input, item_pre_input, timediffs=user_timediffs_tensor, 
                                             features=feature_tensor, adj_embeddings=user_adj_embedding, select='user_update')

            item_pre_emb = model.forward_pre(user_pre_input, item_pre_input, timediffs=item_timediffs_tensor, 
                                             features=feature_tensor, adj_embeddings=item_adj_embedding, select='item_update')
        
        elif args.baseline == 'dgel':
            user_pre_input = user_pre_embedding_input
            item_pre_input = item_pre_embedding_input

            if user_adj is None or item_adj is None or user_timestamp_for_adj is None or item_timestamp_for_adj is None:
                raise ValueError("DGEL evaluation requires user_adj, item_adj, user_timestamp_for_adj, item_timestamp_for_adj.")

            # timestamp_for_adj = timestamp

            user_inter_embeddings, item_inter_embeddings = None, None
            user_adj_embedding, item_adj_embedding = None, None
            user_local_embedding, item_local_embedding = None, None
            user_ext_embeddings, item_ext_embeddings = None, None

            # Update dynamic sub-embeddings based on current interaction
            # Note we only update current user and item, instead of all nodes!
            if len(user_adj[userid]) == 0:
                user_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda()).cuda()
                user_ext_embeddings = Variable(torch.zeros(1, model.final_embedding_dim).cuda()).cuda()
            else:
                user_adj_, user_length_mask, user_max_length = model.adj_sample([user_adj[userid]], model.sample_length)
                user_adj_em = item_pre_embedding[torch.LongTensor(user_adj_).cuda(), :].cuda()

                user_time = [[-1 * (timestamp_for_adj - each) / (timestamp_for_adj - user_timestamp_for_adj[userid][0]) if timestamp_for_adj - user_timestamp_for_adj[userid][0] != 0 else 1 for each in user_timestamp_for_adj[userid]]]
                user_adj_td, _, _ = model.adj_sample(user_time, model.sample_length, 'timediffer')

                # time_decay neighbor GCN
                user_adj_embedding = model.neighbor_aggregate(user_pre_input, user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length, torch.FloatTensor(user_adj_td).cuda(), 'user')
                user_ext_embeddings = model.excitement_aggregate(user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length)

            if len(item_adj[itemid]) == 0:
                item_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda()).cuda()
                item_ext_embeddings = Variable(torch.zeros(1, model.final_embedding_dim).cuda()).cuda()
            else:
                item_adj_, item_length_mask, item_max_length = model.adj_sample([item_adj[itemid]], model.sample_length)
                item_adj_em = user_pre_embedding[torch.LongTensor(item_adj_).cuda(), :].cuda()

                item_time = [[-1 * (timestamp_for_adj - each) / (timestamp_for_adj - item_timestamp_for_adj[itemid][0]) if timestamp_for_adj - item_timestamp_for_adj[itemid][0] != 0 else 1 for each in item_timestamp_for_adj[itemid]]]
                item_adj_td, _, _ = model.adj_sample(item_time, args.sample_length, 'timediffer')

                # time_decay neighbor GCN
                item_adj_embedding = model.neighbor_aggregate(item_pre_input, item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length, torch.FloatTensor(item_adj_td).cuda(), 'item')
                item_ext_embeddings = model.excitement_aggregate(item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length)

            # symbiotic local learning
            user_local_embedding, item_local_embedding = model.local_aggregate(user_pre_input, item_pre_input, user_ext_embeddings, item_ext_embeddings)

            # inherent interaction
            user_inter_embeddings = model.interaction_aggregate(user_pre_input, item_pre_input, feature_tensor, user_timediffs_tensor, 'user')
            item_inter_embeddings = model.interaction_aggregate(item_pre_input, user_pre_input, feature_tensor, item_timediffs_tensor, 'item')

            # forward with re-scaling network
            user_pre_emb = model.forward_pre(user_pre_input, user_inter_embeddings, user_local_embedding, 
                                            timediffs=user_timediffs_tensor, features=feature_tensor, 
                                            adj_embeddings=user_adj_embedding, select='user_update')
            item_pre_emb = model.forward_pre(item_pre_input, item_inter_embeddings, item_local_embedding, 
                                            timediffs=item_timediffs_tensor, features=feature_tensor, 
                                            adj_embeddings=item_adj_embedding, select='item_update')

        elif args.baseline == 'neufilter':
            user_neufilter_emb, user_emb_reg = model.forward_pre(
                user_pre_embedding_input,
                item_pre_embedding_input,
                user_prior=user_projected_embedding,
                users=torch.tensor([userid], dtype=torch.long, device='cuda'),
                features=feature_tensor,
                select='user_update'
            )

            item_neufilter_emb, item_emb_reg = model.forward_pre(
                user_pre_embedding_input,
                item_pre_embedding_input,
                item_prior=item_projected_embedding,
                items=torch.tensor([itemid], dtype=torch.long, device='cuda'),
                features=feature_tensor,
                select='item_update'
            )

            user_pre_emb = user_neufilter_emb
            item_pre_emb = item_neufilter_emb




        # *For post-use preference embedidng*
        sign_tensor = torch.tensor([sign], dtype=torch.float32, device='cuda')    # (1,)
        rating_tensor = torch.tensor(rating, dtype=torch.float32, device='cuda').unsqueeze(0)  # (1, 5)
        current_time_tensor = torch.tensor([timestamp], dtype=torch.float32, device='cuda')  # [1]

        user_post_emb = model.forward_post(user_pos_embedding_input, user_neg_embedding_input,
                                    item_pos_embedding_input, item_neg_embedding_input, sign_tensor, item_pos_embedding, item_neg_embedding, [user_snap], 
                                    select='user_update', timediffs=user_timediffs_tensor, rating=rating_tensor, current_times=current_time_tensor)

        item_post_emb = model.forward_post(item_pos_embedding_input, item_neg_embedding_input,
                                    user_pos_embedding_input, user_neg_embedding_input, sign_tensor, user_pos_embedding, user_neg_embedding, [item_snap], 
                                    select='item_update', timediffs=item_timediffs_tensor, rating=rating_tensor, current_times=current_time_tensor)

        user_post_pos = user_post_emb[:, :args.embedding_dim]
        user_post_neg = user_post_emb[:, args.embedding_dim:]

        item_post_pos = item_post_emb[:, :args.embedding_dim]
        item_post_neg = item_post_emb[:, args.embedding_dim:]

        user_embedding_output = model.forward_fusion(user_pre_emb, user_post_pos, user_post_neg, select='user_update')
        item_embedding_output = model.forward_fusion(item_pre_emb, item_post_pos, item_post_neg, select='item_update')
                
        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 


        
        user_pre_embeddings[userid, :] = user_pre_emb.squeeze(0)
        item_pre_embeddings[itemid, :] = item_pre_emb.squeeze(0)

        user_pos_embeddings[userid, :] = user_post_pos.squeeze(0)
        user_neg_embeddings[userid, :] = user_post_neg.squeeze(0)

        item_pos_embeddings[itemid, :] = item_post_pos.squeeze(0)
        item_neg_embeddings[itemid, :] = item_post_neg.squeeze(0)

        # updatelatest global timestamp
        user_last_ts_global[userid] = timestamp
        item_last_ts_global[itemid] = timestamp

        # DGCF adjacency update
        if args.baseline == 'dgcf' and (args.adj or args.no_zero or args.no_first):
            if not args.sample_length:
                user_adj[userid].add(itemid)
                item_adj[itemid].add(userid)
            else:
                user_adj[userid].append(itemid)
                item_adj[itemid].append(userid)
        elif args.baseline == 'dgel':
            timestamp_for_adj = timedifference_sequence_for_adj[j]

            user_adj[userid].append(itemid)
            item_adj[itemid].append(userid)
            user_timestamp_for_adj[userid].append(timestamp_for_adj)
            item_timestamp_for_adj[itemid].append(timestamp_for_adj)
            if len(user_adj[userid]) > args.sample_length:
                user_adj[userid] = user_adj[userid][-args.sample_length:]
                user_timestamp_for_adj[userid] = user_timestamp_for_adj[userid][-args.sample_length:]
            if len(item_adj[itemid]) > args.sample_length:
                item_adj[itemid] = item_adj[itemid][-args.sample_length:]
                item_timestamp_for_adj[itemid] = item_timestamp_for_adj[itemid][-args.sample_length:]
        
        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        if args.online_test:
            if timestamp - tbatch_start_time > tbatch_timespan:
                tbatch_start_time = timestamp
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                loss = 0 # RESET LOSS FOR NEXT T-BATCH
                item_embeddings.detach_()
                user_embeddings.detach_()

                
                user_pre_embeddings.detach_()
                user_pos_embeddings.detach_()
                user_neg_embeddings.detach_()

                item_pre_embeddings.detach_()
                item_pos_embeddings.detach_()
                item_neg_embeddings.detach_()

                if args.baseline == 'neufilter':
                    model.user_kf.h.detach_()
                    model.item_kf.h.detach_()
        else:
            item_embeddings.detach_()
            user_embeddings.detach_()

            if args.baseline == 'neufilter':
                model.user_kf.h.detach_()
                model.item_kf.h.detach_()

            
            user_pre_embeddings.detach_()
            user_pos_embeddings.detach_()
            user_neg_embeddings.detach_()

            item_pre_embeddings.detach_()
            item_pos_embeddings.detach_()
            item_neg_embeddings.detach_()
        
        torch.cuda.synchronize()
        inference_time_per_epoch += (time.time()-epoch_start_time)
    
    # CALCULATE THE PERFORMANCE METRICS
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    mrr = np.mean([1.0 / r for r in ranks])
    #recall_at_k
    rec1 = sum(np.array(ranks) <= 1)*1.0 / len(ranks)
    rec5 = sum(np.array(ranks) <= 5)*1.0 / len(ranks)
    rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
    rec20 = sum(np.array(ranks) <= 20)*1.0 / len(ranks)
    # precision_at_k
    pre1 = sum(np.array(ranks) <= 1) / (len(ranks) * 1)
    pre5 = sum(np.array(ranks) <= 5) / (len(ranks) * 5)
    pre10 = sum(np.array(ranks) <= 10) / (len(ranks) * 10)
    pre20 = sum(np.array(ranks) <= 20) / (len(ranks) * 20)
    # ndcg_at_k
    ndcg1 = ndcg_at_k(ranks, 1)
    ndcg5 = ndcg_at_k(ranks, 5)
    ndcg10 = ndcg_at_k(ranks, 10)
    ndcg20 = ndcg_at_k(ranks, 20)

    recall = (rec1, rec5, rec10, rec20)
    precision = (pre1, pre5, pre10, pre20)
    ndcg = (ndcg1, ndcg5, ndcg10, ndcg20)
    
    return mrr, recall, precision, ndcg, inference_time_per_epoch, peak_memory_mb


def ndcg_at_k(ranks, k):
    ndcgs = []
    for r in ranks:
        if r <= k:
            dcg = 1.0 / math.log2(r + 1)
            idcg = 1.0 
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
    return np.mean(ndcgs)