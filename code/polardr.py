'''
    python polardr.py --dataset Instant_Video --num_path_u 20 --num_path_i 15 --num_path_u_sample 100 --num_path_i_sample 100 --seed 0 --gpu 0 --project --baseline tracerec  --orth_loss_weight 0.00001


'''
import time
from library_data import *
import library_models as lib
from library_models import *
import datetime
from copy import deepcopy
from evaluate_all_ import eval_one_epoch
from graph import *

from library_jodie import JODIE
from library_dgcf import *
from library_dgel import *
from library_neufilter import NeuFilter


now = datetime.datetime.now()
# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser('polardr')

# select dataset and training mode
parser.add_argument('--dataset', default='Instant_Video', type=str, help='Network name',)
parser.add_argument('--gpu', default=0, type=str, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions {32, 64, ''128'', 256}')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')

# general training hyper-parameters
parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 penalty')  
parser.add_argument('--learning_rate', type=float, default=1e-3, help='l2 penalty')  
parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--span_num', default=500, type=int, help='time span number')
parser.add_argument('--early_stop', type=int, default=5, help='link or link_sign or sign')
parser.add_argument('--tolerance', type=float, default=1e-3, help='tolerated marginal improvement for early stopper')

# options
parser.add_argument('--state_change', action='store_true', help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 
parser.add_argument('--online_test', action='store_true', help='Enable online test mode')

# choose baseline
parser.add_argument('--baseline', type=str, default='tracerec', choices=['tracerec', 'jodie', 'dgcf', 'dgel', 'neufilter'], help='Baseline module used for pre-embedding update')

# method-related hyper-parameters
# temporal walk
parser.add_argument('--walk_type', type=str, default='before', choices=['before', 'point'], help='two type of temporal random walk type')
parser.add_argument('--num_path_u', type=int, default=100, help='number of paths(i.e., alpha1)')
parser.add_argument('--num_path_i', type=int, default=100, help='number of paths(i.e., alpha2)')
parser.add_argument('--len_path', type=int, default=2, help='length of each path(i.e., beta)')
parser.add_argument('--aggregation_method', default='lstm',type=str, help="aggregation_method (ex, GRU, lstm, concat)")
parser.add_argument('--project', action='store_true', help='Enable projection mode (default: false)')
parser.add_argument('--title', type=str, default='none', help='detail method')

# for DGCF, DGEL
# method-related hyper-parameters
parser.add_argument('--adj', default=True, type=bool, help='The second order relationship') # 2hop agg 
parser.add_argument('--no_zero', action='store_true', help='The zero order relationship (default: false)') # interaction aff
parser.add_argument('--no_first', action='store_true', help='The first order relationship (default: false)') # interaction agg 
parser.add_argument('--method', default="gat", help='The way of aggregate adj')
parser.add_argument('--sample_length', type=int, default=None, help='sample length for second order relationship') # [20, 40, 60, 80, 100, 120]
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')
parser.add_argument('--bpr_coefficient', type=float, default=0.001, help='[0.001, 0.0005]') # DGEL


# for NeuFilter
parser.add_argument('--num_layer_kf', default=1, type=int)
parser.add_argument('--reg_factor1', default=0.01, type=float)
parser.add_argument('--reg_factor2', default=0.01, type=float)


# K sampling
parser.add_argument('--num_path_u_sample', type=int, default=100, help='number of paths(i.e., alpha1)')
parser.add_argument('--num_path_i_sample', type=int, default=100, help='number of paths(i.e., alpha2)')


# LOSS weight argument
parser.add_argument('--orth_loss_weight', type=float, default=0.001, help='Weight coefficient for orthogonality loss',)

args = parser.parse_args()

set_random_seed(args.seed)

final_embedding_dim = args.embedding_dim

args.datapath =  "./signed-dataset/%s/%s.csv" % (args.dataset, args.dataset)
log_path = './log/'
best_model_root = './best_models/'
checkpoint_root = './saved_checkpoints/'

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys.argv, now, log_path, checkpoint_root, best_model_root)

if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_pos_timediffs_sequence,
    user_neg_timediffs_sequence, user_previous_itemid_sequence, item_previous_userid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, rating_sequence, sign_sequence,
 user_previous_sign_sequence, item_previous_sign_sequence, timedifference_sequence_for_adj] = load_network(args) #feature: review text

num_interactions = len(user_sequence_id)
num_users = len(user2id) + 1
num_items = len(item2id) + 1

num_features = len(feature_sequence[0]) if len(feature_sequence) > 0 else 0

print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions***\n\n" % (num_users, num_items, num_interactions))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

item_sequence_id_new = [item_id + num_users for item_id in item_sequence_id]
total_nodes = num_users + num_items

full_adj_list = [[] for _ in range(total_nodes)] # for test set
partial_adj_list = [[] for _ in range(total_nodes)] # for train & validation set

for eidx, (src, dst, ts, s) in enumerate(zip(user_sequence_id, item_sequence_id_new, timestamp_sequence, sign_sequence)):
    full_adj_list[src].append((dst, eidx, ts, s))
    full_adj_list[dst].append((src, eidx, ts, s))

for eidx, (src, dst, ts, s) in enumerate(zip(user_sequence_id, item_sequence_id_new, timestamp_sequence, sign_sequence)):
    partial_adj_list[src].append((dst, eidx, ts, s))
    partial_adj_list[dst].append((src, eidx, ts, s))
    if eidx == test_start_idx:
        break

user_adj_snap = [[] for _ in range(num_users)]
item_adj_snap = [[] for _ in range(num_items)]

if args.baseline == 'tracerec':
    full_ngh_finder = NeighborFinder(full_adj_list, walk_type=args.walk_type, bias=0, sample_method='multinomial')
    partial_ngh_finder = NeighborFinder(partial_adj_list, walk_type=args.walk_type,bias=0, sample_method='multinomial')

# SET BATCHING TIMESPAN
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / args.span_num     

# INITIALIZE MODEL AND PARAMETERS
if args.baseline == 'tracerec':
    model = TraceRec(args, num_features, num_users, num_items).cuda()
    model.update_ngh_finder(partial_ngh_finder)

elif args.baseline == 'jodie':
    model = JODIE(args, num_features, num_users, num_items).cuda()
elif args.baseline == 'dgcf':
    model = DGCF(args, num_features, num_users, num_items).cuda()
elif args.baseline == 'dgel':
    model = DGEL(args, num_features, num_users, num_items, final_embedding_dim).cuda()
elif args.baseline == 'neufilter':
    device = torch.device('cuda')
    model = NeuFilter(args, num_features, num_users, num_items, device).cuda()
    
else:
    raise ValueError(f"Unknown baseline: {args.baseline}")

MSELoss = nn.MSELoss()
early_stopper = EarlyStopMonitor(max_round=args.early_stop , tolerance=args.tolerance ) 

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embeddings_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embeddings_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

user_pre_embeddings = initial_user_embedding.repeat(num_users, 1)
user_pos_embeddings = initial_user_embedding.repeat(num_users, 1)
user_neg_embeddings = initial_user_embedding.repeat(num_users, 1)

item_pre_embeddings = initial_item_embedding.repeat(num_items, 1)
item_pos_embeddings = initial_item_embedding.repeat(num_items, 1)
item_neg_embeddings = initial_item_embedding.repeat(num_items, 1)



# INITIALIZE MODEL
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# RUN MODEL
print(f"*** Training baseline={args.baseline} for {args.epochs} epochs ***")

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}
cached_tbatches_timestamp = {}
cached_tbatches_rating = {}
cached_tbatches_sign = {}
mean_learning_time_per_epoch = []
mean_inference_time_per_epoch = []

cached_tbatches_user_pos_timediffs = {}
cached_tbatches_user_neg_timediffs = {}

cached_tbatches_user_adj = {}
cached_tbatches_item_adj = {}
cached_tbatches_user_previous_sign = {}
cached_tbatches_item_previous_sign = {}

cached_tbatches_previous_user = {}

if args.baseline == 'dgel':
    cached_tbatches_user_adj_dgel = {}
    cached_tbatches_item_adj_dgel = {}
    cached_tbatches_user_history_differ = {}
    cached_tbatches_item_history_differ = {}


if args.baseline == 'dgcf':
    cached_tbatches_user_adj_dgcf = {}
    cached_tbatches_item_adj_dgcf = {}
    user_adj = None
    item_adj = None
elif args.baseline == 'dgel':
    user_adj = None
    item_adj = None
    user_timestamp_for_adj = None
    item_timestamp_for_adj = None

for ep in range(args.epochs):

    user_pos_sum = torch.zeros(num_users, device='cuda')
    user_pos_cnt = torch.zeros(num_users, device='cuda')
    user_neg_sum = torch.zeros(num_users, device='cuda')
    user_neg_cnt = torch.zeros(num_users, device='cuda')

    item_pos_sum = torch.zeros(num_items, device='cuda')
    item_pos_cnt = torch.zeros(num_items, device='cuda')
    item_neg_sum = torch.zeros(num_items, device='cuda')
    item_neg_cnt = torch.zeros(num_items, device='cuda')

    user_last_ts_global = np.full(num_users, -1.0, dtype=np.float32)
    item_last_ts_global = np.full(num_items, -1.0, dtype=np.float32)

    optimizer.zero_grad()
    reinitialize_tbatches()
    total_loss, loss, total_interaction_count = 0, 0, 0
    mseloss, ratingloss, dseloss, orthloss = 0, 0, 0, 0
    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False

    # TRAIN TILL THE END OF TRAINING INTERACTION IDX
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    epoch_start_time = time.time()

    if is_first_epoch:
        if args.baseline == 'dgcf':
            if args.adj or args.no_zero or args.no_first:
                if not args.sample_length:
                    user_adj = defaultdict(set)
                    item_adj = defaultdict(set)
                else:
                    user_adj = defaultdict(list)
                    item_adj = defaultdict(list)

        elif args.baseline == 'dgel':
            user_adj = defaultdict(list)
            item_adj = defaultdict(list)
            user_timestamp_for_adj = defaultdict(list)
            item_timestamp_for_adj = defaultdict(list)


    for j in range(train_end_idx):
        if is_first_epoch:
            # READ INTERACTION J
            userid = user_sequence_id[j]
            itemid = item_sequence_id[j]
            feature = feature_sequence[j] if len(feature_sequence) > 0 else []
            rating = rating_sequence[j]
            sign = sign_sequence[j]
            user_timediff = user_timediffs_sequence[j]
            user_pos_timediff = user_pos_timediffs_sequence[j]
            user_neg_timediff = user_neg_timediffs_sequence[j]
            item_timediff = item_timediffs_sequence[j]
            timestamp = timestamp_sequence[j]     

            if args.baseline == 'dgcf':
                if args.adj or args.no_zero or args.no_first:
                
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

                length_of_user = len(user_adj[userid])
                length_of_item = len(item_adj[itemid])
                if length_of_user > args.sample_length:
                    user_adj[userid] = user_adj[userid][length_of_user-args.sample_length:]
                    user_timestamp_for_adj[userid] = user_timestamp_for_adj[userid][length_of_user-args.sample_length:]
                if length_of_item > args.sample_length:
                    item_adj[itemid] = item_adj[itemid][length_of_item-args.sample_length:]
                    item_timestamp_for_adj[itemid] = item_timestamp_for_adj[itemid][length_of_item-args.sample_length:]

            # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
            tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
            lib.tbatchid_user[userid] = tbatch_to_insert
            lib.tbatchid_item[itemid] = tbatch_to_insert

            lib.current_tbatches_user[tbatch_to_insert].append(userid)
            lib.current_tbatches_item[tbatch_to_insert].append(itemid)
            lib.current_tbatches_feature[tbatch_to_insert].append(feature)
            lib.current_tbatches_rating[tbatch_to_insert].append(rating) #rating
            lib.current_tbatches_sign[tbatch_to_insert].append(sign) #sign

            lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
            lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
            lib.current_tbatches_user_pos_timediffs[
                tbatch_to_insert
            ].append(user_pos_timediff)

            lib.current_tbatches_user_neg_timediffs[
                tbatch_to_insert
            ].append(user_neg_timediff)
            lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
            lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
            lib.current_tbatches_previous_user[tbatch_to_insert].append(item_previous_userid_sequence[j])
            lib.current_tbatches_timestamp[tbatch_to_insert].append(timestamp)

            if args.baseline == 'dgcf':
                if args.adj or args.no_zero or args.no_first:
     
                    lib.current_tbatches_user_adj_dgcf[tbatch_to_insert].append(list(user_adj[userid]))  # items
                    lib.current_tbatches_item_adj_dgcf[tbatch_to_insert].append(list(item_adj[itemid]))  # user
            elif args.baseline == 'dgel':
                lib.current_tbatches_user_history_differ[tbatch_to_insert].append(
                    [-1 * (timestamp_for_adj - each) / (timestamp_for_adj - user_timestamp_for_adj[userid][0])
                    if timestamp_for_adj - user_timestamp_for_adj[userid][0] != 0 else 1
                    for each in user_timestamp_for_adj[userid]]
                )

                lib.current_tbatches_item_history_differ[tbatch_to_insert].append(
                    [-1 * (timestamp_for_adj - each) / (timestamp_for_adj - item_timestamp_for_adj[itemid][0])
                    if timestamp_for_adj - item_timestamp_for_adj[itemid][0] != 0 else 1
                    for each in item_timestamp_for_adj[itemid]]
                )

                lib.current_tbatches_user_adj_dgel[tbatch_to_insert].append(deepcopy(user_adj[userid]))
                lib.current_tbatches_item_adj_dgel[tbatch_to_insert].append(deepcopy(item_adj[itemid]))

            user_adj_snap[userid].append((itemid, j, timestamp_sequence[j], sign_sequence[j]))
            item_adj_snap[itemid].append((userid, j, timestamp_sequence[j], sign_sequence[j]))
            

            
            if len(user_adj_snap[userid]) > args.num_path_u_sample:
                user_adj_snap[userid] = user_adj_snap[userid][-args.num_path_u_sample:]
            if len(item_adj_snap[itemid]) > args.num_path_i_sample:
                item_adj_snap[itemid] = item_adj_snap[itemid][-args.num_path_i_sample:]

            user_snap_with_last = [(nb, eidx, ts, s, item_last_ts_global[nb]) for nb, eidx, ts, s in user_adj_snap[userid]]
            item_snap_with_last = [(nb, eidx, ts, s, user_last_ts_global[nb]) for nb, eidx, ts, s in item_adj_snap[itemid]]

            lib.current_tbatches_user_adj[tbatch_to_insert].append(
                user_snap_with_last
            )
            lib.current_tbatches_item_adj[tbatch_to_insert].append(
                item_snap_with_last
            )
            # global last timestamp update
            user_last_ts_global[userid] = timestamp_sequence[j]
            item_last_ts_global[itemid] = timestamp_sequence[j]

            lib.current_tbatches_user_previous_sign[tbatch_to_insert].append(user_previous_sign_sequence[j])
            lib.current_tbatches_item_previous_sign[tbatch_to_insert].append(item_previous_sign_sequence[j])
            
        timestamp = timestamp_sequence[j]
        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # ITERATE OVER ALL T-BATCHES
            if not is_first_epoch:
                lib.current_tbatches_user = cached_tbatches_user[timestamp]
                lib.current_tbatches_item = cached_tbatches_item[timestamp]
                lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                lib.current_tbatches_rating = cached_tbatches_rating[timestamp] #rating
                lib.current_tbatches_sign = cached_tbatches_sign[timestamp] #sign

                lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]

                lib.current_tbatches_user_pos_timediffs = (
                    cached_tbatches_user_pos_timediffs[timestamp]
                )

                lib.current_tbatches_user_neg_timediffs = (
                    cached_tbatches_user_neg_timediffs[timestamp]
                )

                lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]
                lib.current_tbatches_previous_user = cached_tbatches_previous_user[timestamp]
                lib.current_tbatches_timestamp = cached_tbatches_timestamp[timestamp]

                lib.current_tbatches_user_adj = cached_tbatches_user_adj[timestamp] 
                lib.current_tbatches_item_adj = cached_tbatches_item_adj[timestamp] 

                lib.current_tbatches_user_previous_sign = cached_tbatches_user_previous_sign[timestamp]
                lib.current_tbatches_item_previous_sign = cached_tbatches_item_previous_sign[timestamp]

                if args.baseline == 'dgcf':
                    lib.current_tbatches_user_adj_dgcf = cached_tbatches_user_adj_dgcf[timestamp]
                    lib.current_tbatches_item_adj_dgcf = cached_tbatches_item_adj_dgcf[timestamp]
                elif args.baseline == 'dgel':
                    lib.current_tbatches_user_adj_dgel = cached_tbatches_user_adj_dgel[timestamp]
                    lib.current_tbatches_item_adj_dgel = cached_tbatches_item_adj_dgel[timestamp]
                    lib.current_tbatches_user_history_differ = cached_tbatches_user_history_differ[timestamp]
                    lib.current_tbatches_item_history_differ = cached_tbatches_item_history_differ[timestamp]

            for i in range(len(lib.current_tbatches_user)):
                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                # LOAD THE CURRENT TBATCH
                if is_first_epoch:
                    lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                    lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                    lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                    lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()
                    lib.current_tbatches_rating[i] = torch.Tensor(np.array(lib.current_tbatches_rating[i])).cuda()
                    lib.current_tbatches_sign[i] = torch.Tensor(lib.current_tbatches_sign[i]).cuda() #sign

                    lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                    lib.current_tbatches_user_pos_timediffs[i] = torch.Tensor(
                        lib.current_tbatches_user_pos_timediffs[i]
                    ).cuda()

                    lib.current_tbatches_user_neg_timediffs[i] = torch.Tensor(
                        lib.current_tbatches_user_neg_timediffs[i]
                    ).cuda()
                    lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                    lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                    lib.current_tbatches_previous_user[i] = torch.LongTensor(lib.current_tbatches_previous_user[i]).cuda()
                    lib.current_tbatches_timestamp[i] = torch.Tensor(lib.current_tbatches_timestamp[i]).cuda()

                    lib.current_tbatches_user_previous_sign[i] = torch.Tensor(lib.current_tbatches_user_previous_sign[i]).cuda()
                    lib.current_tbatches_item_previous_sign[i] = torch.Tensor(lib.current_tbatches_item_previous_sign[i]).cuda()

                tbatch_userids = lib.current_tbatches_user[i] 
                tbatch_itemids = lib.current_tbatches_item[i] 
                tbatch_interactionids = lib.current_tbatches_interactionids[i]
                feature_tensor = lib.current_tbatches_feature[i] 
                rating = lib.current_tbatches_rating[i] #rating
                sign = lib.current_tbatches_sign[i] #sign

                user_timediffs_tensor = lib.current_tbatches_user_timediffs[i].unsqueeze(1)
                user_pos_timediffs_tensor = (
                    lib.current_tbatches_user_pos_timediffs[i].unsqueeze(1)
                )

                user_neg_timediffs_tensor = (
                    lib.current_tbatches_user_neg_timediffs[i].unsqueeze(1)
                )
                item_timediffs_tensor = lib.current_tbatches_item_timediffs[i].unsqueeze(1)
                tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                item_embedding_previous = item_embeddings[tbatch_itemids_previous,:].detach()
                
                tbatch_userids_previous = lib.current_tbatches_previous_user[i]
                user_embedding_previous = user_embeddings[tbatch_userids_previous,:].detach()

                tbatch_times = lib.current_tbatches_timestamp[i]

                if args.baseline == 'tracerec':
                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input = user_embeddings[tbatch_userids,:]
                    user_projected_embedding = model.forward_project(user_embedding_input, user_timediffs_tensor)
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[tbatch_itemids_previous,:], user_embeddings_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING                            
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    mseloss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embeddings_static[tbatch_itemids,:]], dim=1).detach())

                    user_pre_embedding = user_pre_embeddings
                    user_pos_embedding = user_pos_embeddings
                    user_neg_embedding = user_neg_embeddings

                    item_pre_embedding = item_pre_embeddings
                    item_pos_embedding = item_pos_embeddings
                    item_neg_embedding = item_neg_embeddings
            
                    # *For pre-use preference embedidng*
                    # load subgraph for temporal random walk
                    temporal_path_user = model.grab_subgraph(tbatch_userids, tbatch_times, args.len_path, args.num_path_u)
                    temporal_path_item = model.grab_subgraph(tbatch_itemids + num_users, tbatch_times, args.len_path, args.num_path_i)

                    # (M1) Output: path
                    temporal_path_user = model.subgraph_tree2walk(tbatch_userids.cpu().numpy(), tbatch_times.cpu().numpy(), temporal_path_user)
                    temporal_path_item = model.subgraph_tree2walk(tbatch_itemids.cpu().numpy() + num_users, tbatch_times.cpu().numpy(), temporal_path_item)

                    # (M2) Output: restored embedding
                    u_neighbor_records, u_time_decay_weights, u_masks, u_time_feat = model.forward_msg_time_delta(temporal_path_user) 
                    i_neighbor_records, i_time_decay_weights, i_masks, i_time_feat = model.forward_msg_time_delta(temporal_path_item)

                    u_final_embeddings, u_valid_mask = model.get_valid_embeddings_vectorized(u_neighbor_records, u_time_feat, u_masks, user_pre_embedding, item_pre_embedding, 'user')
                    i_final_embeddings, i_valid_mask = model.get_valid_embeddings_vectorized(i_neighbor_records, i_time_feat, i_masks, item_pre_embedding, user_pre_embedding, 'item')
                    
                    # (M3) Output: embedding
                    u_walk_final_embeddings = model.aggregate_embeddigs(u_final_embeddings, u_valid_mask, u_time_decay_weights, \
                                                                        aggregation_method=args.aggregation_method, project=args.project)
                    i_walk_final_embeddings = model.aggregate_embeddigs(i_final_embeddings, i_valid_mask, i_time_decay_weights, \
                                                                        aggregation_method=args.aggregation_method, project=args.project)

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    # user embedding update (dst: item)
                    user_pre_emb = model.forward_pre(dst_pre = item_pre_embedding[tbatch_itemids,:],\
                                                    historical_embeddings = u_walk_final_embeddings, timediffs = user_timediffs_tensor)
                    # item embedding update (dst: user)
                    item_pre_emb = model.forward_pre(dst_pre = user_pre_embedding[tbatch_userids,:],\
                                                    historical_embeddings = i_walk_final_embeddings, timediffs=item_timediffs_tensor)
                elif args.baseline == 'jodie':
                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input = user_embeddings[tbatch_userids,:]
                    user_projected_embedding = model.forward_pre(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[tbatch_itemids_previous,:], user_embeddings_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING                            
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    mseloss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embeddings_static[tbatch_itemids,:]], dim=1).detach())

                    user_pre_embedding = user_pre_embeddings
                    user_pos_embedding = user_pos_embeddings
                    user_neg_embedding = user_neg_embeddings

                    item_pre_embedding = item_pre_embeddings
                    item_pos_embedding = item_pos_embeddings
                    item_neg_embedding = item_neg_embeddings

                    # current tbatch pre embeddings
                    user_pre_input = user_pre_embedding[tbatch_userids, :]
                    item_pre_input = item_pre_embedding[tbatch_itemids, :]

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_pre_emb = model.forward_pre(user_pre_input, item_pre_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                    item_pre_emb = model.forward_pre(user_pre_input, item_pre_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                
                elif args.baseline == 'dgcf':
                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input = user_embeddings[tbatch_userids,:]
                    user_projected_embedding = model.forward_pre(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[tbatch_itemids_previous,:], user_embeddings_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    mseloss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embeddings_static[tbatch_itemids,:]], dim=1).detach())

                    
                    user_pre_embedding = user_pre_embeddings
                    user_pos_embedding = user_pos_embeddings
                    user_neg_embedding = user_neg_embeddings

                    item_pre_embedding = item_pre_embeddings
                    item_pos_embedding = item_pos_embeddings
                    item_neg_embedding = item_neg_embeddings

                    # current tbatch pre embeddings
                    user_pre_input = user_pre_embedding[tbatch_userids, :]
                    item_pre_input = item_pre_embedding[tbatch_itemids, :]

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    if args.adj or args.no_zero or args.no_first:
                        if not args.sample_length:
                            user_adj_, user_length_mask, user_max_length = adj_pad(lib.current_tbatches_user_adj_dgcf[i])
                            item_adj_, item_length_mask, item_max_length = adj_pad(lib.current_tbatches_item_adj_dgcf[i])
                        else:
                            user_adj_, user_length_mask, user_max_length = adj_sample(lib.current_tbatches_user_adj_dgcf[i],
                                                                                args.sample_length)
                            item_adj_, item_length_mask, item_max_length = adj_sample(lib.current_tbatches_item_adj_dgcf[i],
                                                                                args.sample_length)
                        user_adj_em = item_pre_embedding[torch.LongTensor(user_adj_).cuda(), :]
                        item_adj_em = user_pre_embedding[torch.LongTensor(item_adj_).cuda(), :]
                        
                        
                        if model.method == 'mean':
                            user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_pre_input,
                                                                    select='user_update')
                            item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_pre_input,
                                                                    select='item_update')
                        elif model.method == 'attention':
                            user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                                        user_max_length, user_pre_input,
                                                                        select='user_update')
                            item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                        item_max_length, item_pre_input,
                                                                        select='item_update')
                        elif model.method == 'gat':
                            user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                                        user_max_length, user_pre_input,
                                                                        select='user_update')
                            item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                                        item_max_length, item_pre_input,
                                                                        select='item_update')
                        elif model.method == 'lstm':
                            user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_pre_input,
                                                                    select='user_update')
                            item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_pre_input,
                                                                    select='item_update')
                    else:
                        user_adj_embedding = None
                        item_adj_embedding = None
                    

                    user_pre_emb = model.forward_pre(user_pre_input, item_pre_input, timediffs=user_timediffs_tensor, features=feature_tensor, adj_embeddings=user_adj_embedding, select='user_update')
                    item_pre_emb = model.forward_pre(user_pre_input, item_pre_input, timediffs=item_timediffs_tensor, features=feature_tensor, adj_embeddings=item_adj_embedding, select='item_update')

                elif args.baseline == 'dgel':
                    user_embedding_input = user_embeddings[tbatch_userids, :]
                    item_embedding_input = item_embeddings[tbatch_itemids, :]

                    # future prediction
                    user_projected_embedding = model.forward_pre(user_embedding_input, None, None, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[tbatch_itemids_previous, :], user_embeddings_static[tbatch_userids, :]], dim=1).cuda()
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    mseloss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embeddings_static[tbatch_itemids,:]], dim=1).detach())

                    user_pre_embedding = user_pre_embeddings
                    user_pos_embedding = user_pos_embeddings
                    user_neg_embedding = user_neg_embeddings

                    item_pre_embedding = item_pre_embeddings
                    item_pos_embedding = item_pos_embeddings
                    item_neg_embedding = item_neg_embeddings

                    user_pre_input = user_pre_embedding[tbatch_userids, :]
                    item_pre_input = item_pre_embedding[tbatch_itemids, :]

                    user_adj_, user_length_mask, user_max_length = model.adj_sample(lib.current_tbatches_user_adj_dgel[i], args.sample_length)
                    item_adj_, item_length_mask, item_max_length = model.adj_sample(lib.current_tbatches_item_adj_dgel[i], args.sample_length)

                
                    user_adj_td, _, _ = model.adj_sample(lib.current_tbatches_user_history_differ[i], args.sample_length, 'timediffer')
                    item_adj_td, _, _ = model.adj_sample(lib.current_tbatches_item_history_differ[i], args.sample_length, 'timediffer')


                    user_adj_em = item_pre_embedding[torch.LongTensor(user_adj_).cuda(), :]
                    item_adj_em = user_pre_embedding[torch.LongTensor(item_adj_).cuda(), :]


                    # 1: inherent interaction
                    user_inter_embeddings = model.interaction_aggregate(user_pre_input, item_pre_input, feature_tensor, user_timediffs_tensor, 'user')
                    item_inter_embeddings = model.interaction_aggregate(item_pre_input, user_pre_input, feature_tensor, item_timediffs_tensor, 'item')


                    # 2: time-decay neighbor GCN
                    user_adj_embedding = model.neighbor_aggregate(user_pre_input, user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length, torch.FloatTensor(user_adj_td).cuda(), 'user')
                    item_adj_embedding = model.neighbor_aggregate(item_pre_input, item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length, torch.FloatTensor(item_adj_td).cuda(), 'item')


                    # 3: symbiotic local learning
                    user_ext_embeddings = model.excitement_aggregate(user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length)
                    item_ext_embeddings = model.excitement_aggregate(item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length)
                    user_local_embedding, item_local_embedding = model.local_aggregate(user_pre_input, item_pre_input, user_ext_embeddings, item_ext_embeddings)



                    user_pre_emb = model.forward_pre(user_pre_input, user_inter_embeddings, user_local_embedding,
                                        timediffs=user_timediffs_tensor, features=feature_tensor,
                                        adj_embeddings=user_adj_embedding, select='user_update')

                    item_pre_emb = model.forward_pre(item_pre_input, item_inter_embeddings, item_local_embedding,
                                        timediffs=item_timediffs_tensor, features=feature_tensor,
                                        adj_embeddings=item_adj_embedding, select='item_update')
                
                elif args.baseline == 'neufilter':
                    user_embedding_input = user_embeddings[tbatch_userids, :]
                    item_embedding_input = item_embeddings[tbatch_itemids, :]

                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_projected_embedding, item_projected_embedding = model.forward_pre(user_embedding_input, item_embedding_input, user_timediffs=user_timediffs_tensor, item_timediffs=item_timediffs_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[tbatch_itemids_previous,:], user_embeddings_static[tbatch_userids,:]], dim=1)


                    # PREDICT NEXT ITEM EMBEDDING
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    mseloss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embeddings_static[tbatch_itemids, :]], dim=1).detach())


                    user_pre_embedding = user_pre_embeddings
                    user_pos_embedding = user_pos_embeddings
                    user_neg_embedding = user_neg_embeddings

                    item_pre_embedding = item_pre_embeddings
                    item_pos_embedding = item_pos_embeddings
                    item_neg_embedding = item_neg_embeddings

                    user_pre_input = user_pre_embedding[tbatch_userids, :]
                    item_pre_input = item_pre_embedding[tbatch_itemids, :]

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_neufilter_emb, user_emb_reg = model.forward_pre(user_pre_input, item_pre_input, user_prior=user_projected_embedding, users=tbatch_userids, features=feature_tensor, select='user_update')
                    item_neufilter_emb, item_emb_reg = model.forward_pre(user_pre_input, item_pre_input, item_prior=item_projected_embedding, items=tbatch_itemids, features=feature_tensor, select='item_update')

                    # Treat NeuFilter output as pre_embedding for your framework
                    user_pre_emb = user_neufilter_emb
                    item_pre_emb = item_neufilter_emb


                # *For post-use preference embedidng*
                # tgt: user, dst: item
                # output: pos||neg
                user_post_emb = model.forward_post(user_pos_embedding[tbatch_userids, :],user_neg_embedding[tbatch_userids, :], 
                                        item_pos_embedding[tbatch_itemids, :], item_neg_embedding[tbatch_itemids, :], sign, item_pos_embedding, item_neg_embedding, lib.current_tbatches_user_adj[i], 
                                        select='user_update', timediffs=user_timediffs_tensor, rating=rating, current_times=tbatch_times)
                item_post_emb = model.forward_post(item_pos_embedding[tbatch_itemids, :], item_neg_embedding[tbatch_itemids, :],
                                        user_pos_embedding[tbatch_userids, :], user_neg_embedding[tbatch_userids, :], sign, user_pos_embedding, user_neg_embedding, lib.current_tbatches_item_adj[i], 
                                        select='item_update', timediffs=item_timediffs_tensor, rating=rating, current_times=tbatch_times)

                user_post_pos = user_post_emb[:, :args.embedding_dim]
                user_post_neg = user_post_emb[:, args.embedding_dim:]

                item_post_pos = item_post_emb[:, :args.embedding_dim]
                item_post_neg = item_post_emb[:, args.embedding_dim:]

                # residual polarity orhtloss
                user_orthloss = residual_orthogonality_loss(pre=user_pre_emb, pos=user_post_pos, neg=user_post_neg,)
                item_orthloss = residual_orthogonality_loss(pre=item_pre_emb, pos=item_post_pos, neg=item_post_neg,)\

                orthloss += 1 * (user_orthloss + item_orthloss)

                user_embedding_output = model.forward_fusion(user_pre_emb, user_post_pos, user_post_neg, select='user_update')
                item_embedding_output = model.forward_fusion(item_pre_emb, item_post_pos, item_post_neg, select='item_update')
                
                
                user_embeddings = user_embeddings.index_copy(0, tbatch_userids, user_embedding_output)
                item_embeddings = item_embeddings.index_copy(0, tbatch_itemids, item_embedding_output)

                user_pre_embeddings = user_pre_embeddings.index_copy(0, tbatch_userids, user_pre_emb)
                item_pre_embeddings = item_pre_embeddings.index_copy(0, tbatch_itemids, item_pre_emb)

                user_pos_embeddings = user_pos_embeddings.index_copy(0, tbatch_userids, user_post_pos)
                user_neg_embeddings = user_neg_embeddings.index_copy(0, tbatch_userids, user_post_neg)

                item_pos_embeddings = item_pos_embeddings.index_copy(0, tbatch_itemids, item_post_pos)
                item_neg_embeddings = item_neg_embeddings.index_copy(0, tbatch_itemids, item_post_neg)


                loss = mseloss + args.orth_loss_weight * orthloss
            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = 0 # RESET LOSS FOR NEXT T-BATCH
            mseloss = 0 # RESET LOSS FOR NEXT T-BATCH
            dseloss = 0 # RESET LOSS FOR NEXT T-BATCH
            orthloss = 0 # RESET LOSS FOR NEXT T-BATCH
            ratingloss = 0 # RESET LOSS FOR NEXT T-BATCH
            user_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
            item_embeddings.detach_()

            user_pre_embeddings.detach_()
            user_pos_embeddings.detach_()
            user_neg_embeddings.detach_()

            item_pre_embeddings.detach_()
            item_pos_embeddings.detach_()
            item_neg_embeddings.detach_()

            if args.baseline == 'neufilter':
                model.user_kf.h.detach_()
                model.item_kf.h.detach_()
            
            # REINITIALIZE
            if is_first_epoch:
                cached_tbatches_user[timestamp] = lib.current_tbatches_user
                cached_tbatches_item[timestamp] = lib.current_tbatches_item
                cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                cached_tbatches_user_pos_timediffs[timestamp] = (
                    lib.current_tbatches_user_pos_timediffs
                )

                cached_tbatches_user_neg_timediffs[timestamp] = (
                    lib.current_tbatches_user_neg_timediffs
                )
                cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item
                cached_tbatches_previous_user[timestamp] = lib.current_tbatches_previous_user
                cached_tbatches_timestamp[timestamp] = lib.current_tbatches_timestamp
                cached_tbatches_rating[timestamp] = lib.current_tbatches_rating #rating
                cached_tbatches_sign[timestamp] = lib.current_tbatches_sign #sign
                cached_tbatches_user_adj[timestamp] = lib.current_tbatches_user_adj 
                cached_tbatches_item_adj[timestamp] = lib.current_tbatches_item_adj 
                cached_tbatches_user_previous_sign[timestamp] = lib.current_tbatches_user_previous_sign 
                cached_tbatches_item_previous_sign[timestamp] = lib.current_tbatches_item_previous_sign 
                
                if args.baseline == 'dgcf':
                    cached_tbatches_user_adj_dgcf[timestamp] = lib.current_tbatches_user_adj_dgcf
                    cached_tbatches_item_adj_dgcf[timestamp] = lib.current_tbatches_item_adj_dgcf

                elif args.baseline == 'dgel':
                    cached_tbatches_user_adj_dgel[timestamp] = lib.current_tbatches_user_adj_dgel
                    cached_tbatches_item_adj_dgel[timestamp] = lib.current_tbatches_item_adj_dgel
                    cached_tbatches_user_history_differ[timestamp] = lib.current_tbatches_user_history_differ
                    cached_tbatches_item_history_differ[timestamp] = lib.current_tbatches_item_history_differ

                reinitialize_tbatches()
                tbatch_to_insert = -1

    is_first_epoch = False # as first epoch ends here
    if ep == 0:
        del user_adj_snap, item_adj_snap 
    torch.cuda.synchronize()
    learning_time_per_epoch = time.time()-epoch_start_time

    train_peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) # memory check
    train_current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    if ep > 0:  
        mean_learning_time_per_epoch.append(learning_time_per_epoch)

    val_user_adj = deepcopy(user_adj) if args.baseline in ['dgcf', 'dgel'] else None
    val_item_adj = deepcopy(item_adj) if args.baseline in ['dgcf', 'dgel'] else None

    val_user_timestamp_for_adj = deepcopy(user_timestamp_for_adj) if args.baseline == 'dgel' else None
    val_item_timestamp_for_adj = deepcopy(item_timestamp_for_adj) if args.baseline == 'dgel' else None


    val_mrr, val_recall, val_precision, val_ndcg, inference_time_per_epoch, val_peak_memory_mb \
        = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, rating_sequence, sign_sequence,
                   user_timediffs_sequence, user_pos_timediffs_sequence, user_neg_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, item_previous_userid_sequence,
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static,
                   validation_start_idx, test_start_idx, num_users, partial_adj_list,
                   user_previous_sign_sequence, item_previous_sign_sequence,
                   user_pos_sum, user_pos_cnt, user_neg_sum, user_neg_cnt,
                    item_pos_sum, item_pos_cnt, item_neg_sum, item_neg_cnt,
                    user_adj=val_user_adj,
                    item_adj=val_item_adj,
                    user_timestamp_for_adj=val_user_timestamp_for_adj,
                    item_timestamp_for_adj=val_item_timestamp_for_adj,
                    timedifference_sequence_for_adj=timedifference_sequence_for_adj if args.baseline == 'dgel' else None,
                    user_pre_embeddings=user_pre_embeddings,
                    user_pos_embeddings=user_pos_embeddings,
                    user_neg_embeddings=user_neg_embeddings,
                    item_pre_embeddings=item_pre_embeddings,
                    item_pos_embeddings=item_pos_embeddings,
                    item_neg_embeddings=item_neg_embeddings)

    if args.baseline in ['dgcf', 'dgel']:
        test_user_adj = val_user_adj
        test_item_adj = val_item_adj

    if args.baseline == 'dgel':
        test_user_timestamp_for_adj = val_user_timestamp_for_adj
        test_item_timestamp_for_adj = val_item_timestamp_for_adj

    mean_inference_time_per_epoch.append(inference_time_per_epoch)
   
    val_rec1, val_rec5, val_rec10, val_rec20 = val_recall
    val_pre1, val_pre5, val_pre10, val_pre20 = val_precision
    val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20 = val_ndcg

    # End of epoch
    logger.info('epoch: {}'.format(ep))
    logger.info('Total loss in this epoch: {}'.format(total_loss))
    logger.info('val MRR:\t{:.4f}'.format(val_mrr))
    logger.info('val REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_rec1, val_rec5, val_rec10, val_rec20))
    logger.info('val PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_pre1, val_pre5, val_pre10, val_pre20))
    logger.info('val NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20))
    logger.info('Trainig time: {} sec'.format(learning_time_per_epoch))
    logger.info('Inference time: {} sec'.format(inference_time_per_epoch))
    logger.info('Training peak memory: {:.2f} MiB (current: {:.2f} MiB)'.format(train_peak_memory_mb, train_current_memory_mb))  
    logger.info('Inference peak memory: {:.2f} MiB'.format(val_peak_memory_mb))  


    if early_stopper.early_stop_check(val_mrr):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)

        model, optimizer, user_embeddings_dystat, item_embeddings_dystat, mu_stats, independent_states = load_model(model, optimizer, args, best_checkpoint_path)

        item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
        item_embeddings = item_embeddings.clone()
        item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
        item_embeddings_static = item_embeddings_static.clone()

        user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
        user_embeddings = user_embeddings.clone()
        user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
        user_embeddings_static = user_embeddings_static.clone()
        
        user_pos_sum = mu_stats['user_pos_sum']
        user_pos_cnt = mu_stats['user_pos_cnt']
        user_neg_sum = mu_stats['user_neg_sum']
        user_neg_cnt = mu_stats['user_neg_cnt']
        item_pos_sum = mu_stats['item_pos_sum']
        item_pos_cnt = mu_stats['item_pos_cnt']
        item_neg_sum = mu_stats['item_neg_sum']
        item_neg_cnt = mu_stats['item_neg_cnt']

        if independent_states is None:
            raise ValueError(
                "Checkpoint does not contain pre/pos/neg embedding states."
            )

        user_pre_embeddings = independent_states['user_pre_embeddings'].clone()
        user_pos_embeddings = independent_states['user_pos_embeddings'].clone()
        user_neg_embeddings = independent_states['user_neg_embeddings'].clone()

        item_pre_embeddings = independent_states['item_pre_embeddings'].clone()
        item_pos_embeddings = independent_states['item_pos_embeddings'].clone()
        item_neg_embeddings = independent_states['item_neg_embeddings'].clone()

        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        model.eval()
        break
    else:
        item_embeddings_dystat = torch.cat([item_embeddings, item_embeddings_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embeddings_static], dim=1)

        mu_stats = {
            'user_pos_sum': user_pos_sum, 'user_pos_cnt': user_pos_cnt,
            'user_neg_sum': user_neg_sum, 'user_neg_cnt': user_neg_cnt,
            'item_pos_sum': item_pos_sum, 'item_pos_cnt': item_pos_cnt,
            'item_neg_sum': item_neg_sum, 'item_neg_cnt': item_neg_cnt,
        }

        independent_states = {
            'user_pre_embeddings': user_pre_embeddings,
            'user_pos_embeddings': user_pos_embeddings,
            'user_neg_embeddings': user_neg_embeddings,
            'item_pre_embeddings': item_pre_embeddings,
            'item_pos_embeddings': item_pos_embeddings,
            'item_neg_embeddings': item_neg_embeddings,
        }

        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, get_checkpoint_path(ep), mu_stats=mu_stats, independent_states=independent_states)
        
        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

        user_pre_embeddings = initial_user_embedding.repeat(num_users, 1)
        user_pos_embeddings = initial_user_embedding.repeat(num_users, 1)
        user_neg_embeddings = initial_user_embedding.repeat(num_users, 1)

        item_pre_embeddings = initial_item_embedding.repeat(num_items, 1)
        item_pos_embeddings = initial_item_embedding.repeat(num_items, 1)
        item_neg_embeddings = initial_item_embedding.repeat(num_items, 1)


# Save final model of final epoch
logger.info("\n***** Training complete. Testing final model. *****\n")
if args.baseline == 'tracerec':
    model.update_ngh_finder(full_ngh_finder)


test_mrr, test_recall, test_precision, test_ndcg, inference_time, test_peak_memory_mb \
        = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, rating_sequence, sign_sequence,
                   user_timediffs_sequence, user_pos_timediffs_sequence, user_neg_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, item_previous_userid_sequence,
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static,
                   test_start_idx, test_end_idx, num_users, full_adj_list,
                   user_previous_sign_sequence, item_previous_sign_sequence,
                   user_pos_sum, user_pos_cnt, user_neg_sum, user_neg_cnt,
                    item_pos_sum, item_pos_cnt, item_neg_sum, item_neg_cnt,
                    user_adj=test_user_adj if args.baseline in ['dgcf', 'dgel'] else None,
                    item_adj=test_item_adj if args.baseline in ['dgcf', 'dgel'] else None,
                    user_timestamp_for_adj=test_user_timestamp_for_adj if args.baseline == 'dgel' else None,
                    item_timestamp_for_adj=test_item_timestamp_for_adj if args.baseline == 'dgel' else None,
                    timedifference_sequence_for_adj=timedifference_sequence_for_adj if args.baseline == 'dgel' else None,
                    user_pre_embeddings=user_pre_embeddings,
                    user_pos_embeddings=user_pos_embeddings,
                    user_neg_embeddings=user_neg_embeddings,
                    item_pre_embeddings=item_pre_embeddings,
                    item_pos_embeddings=item_pos_embeddings,
                    item_neg_embeddings=item_neg_embeddings)
                    
test_rec1, test_rec5, test_rec10, test_rec20 = test_recall
test_pre1, test_pre5, test_pre10, test_pre20 = test_precision
test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20 = test_ndcg

logger.info('Best epoch: {}'.format(early_stopper.best_epoch))
logger.info('test MRR:\t{:.4f}'.format(test_mrr))
logger.info('test REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_rec1, test_rec5, test_rec10, test_rec20))
logger.info('test PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_pre1, test_pre5, test_pre10, test_pre20))
logger.info('test NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20))
logger.info('Average training time per epoch: {}'.format(np.mean(mean_learning_time_per_epoch)))
logger.info('Average inference time per epoch: {}'.format(np.mean(mean_inference_time_per_epoch)))
logger.info('Inference time (testing): {} sec'.format(inference_time))
logger.info('Inference peak memory: {:.2f} MiB'.format(test_peak_memory_mb)) 

torch.save(model.state_dict(), best_model_path)
