from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from sklearn.preprocessing import scale
import torch
import logging

# LOAD THE DATASET
def load_network(args, time_scaling=True):
    dataset = args.dataset
    datapath = args.datapath

    is_kuairand = dataset.lower() == "kuairand_pure"

    user_sequence = []
    item_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    rating_sequence = []
    start_timestamp = None

    print("\n\n**** Loading %s network from file: %s ****" % (dataset, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f): # FORMAT: user_id,item_id,rating,timestamp
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        rating_sequence.append(float(ls[2]))
        if start_timestamp is None:
            start_timestamp = float(ls[3])
        timestamp_sequence.append(float(ls[3]) - start_timestamp)
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    rating_sequence = np.array(rating_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    if is_kuairand:
        # KuaiRand:
        # is_click == 1 -> positive
        # is_click == 0 -> negative
        invalid_values = np.setdiff1d(
            np.unique(rating_sequence),
            np.array([0.0, 1.0])
        )

        if len(invalid_values) > 0:
            raise ValueError(
                f"KuaiRand is_click must contain only 0 and 1. "
                f"Invalid values found: {invalid_values}"
            )

        sign_sequence = np.where(
            rating_sequence == 1,
            1,
            -1
        )
    else:
        # rating >= 4 -> positive
        # rating <= 3 -> negative
        sign_sequence = np.where(rating_sequence >= 4, 1, -1) 

    print("Formating item sequence")
    nodeid = 1
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 1
    user2id = {}

    # last/positive/negative interaction timestamp
    user_last_timestamp = {}
    user_last_pos_timestamp = {} 
    user_last_neg_timestamp = {} 

    user_timedifference_sequence = []
    user_pos_timedifference_sequence = [] 
    user_neg_timedifference_sequence = []


    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: 0)

    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1

        timestamp = timestamp_sequence[cnt]
        sign = sign_sequence[cnt]

        if user in user_last_timestamp:
            pre_delta = timestamp - user_last_timestamp[user]
        else:
            pre_delta = timestamp

        if user in user_last_pos_timestamp:
            pos_delta = timestamp - user_last_pos_timestamp[user]
        else:
            pos_delta = pre_delta

        if user in user_last_neg_timestamp:
            neg_delta = timestamp - user_last_neg_timestamp[user]
        else:
            neg_delta = pre_delta

        user_timedifference_sequence.append(pre_delta)
        user_pos_timedifference_sequence.append(pos_delta)
        user_neg_timedifference_sequence.append(neg_delta)


        user_last_timestamp[user] = timestamp

        if sign == 1:
            user_last_pos_timestamp[user] = timestamp
        elif sign == -1:
            user_last_neg_timestamp[user] = timestamp

        user_previous_itemid_sequence.append(
            user_latest_itemid[user]
        )
        user_latest_itemid[user] = item2id[item_sequence[cnt]]  
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    print("Formating item previous user sequence")
    item_previous_userid_sequence = []
    item_latest_userid = defaultdict(lambda: 0)

    for cnt, item in enumerate(item_sequence):
        item_id = item2id[item]
        user_id = user_sequence_id[cnt]
        item_previous_userid_sequence.append(item_latest_userid[item_id])
        item_latest_userid[item_id] = user_id

    print("Formating previous sign sequences")
    user_previous_sign_sequence = []
    user_latest_sign = defaultdict(lambda: 0)
    for cnt, user in enumerate(user_sequence):
        user_id = user_sequence_id[cnt]
        user_previous_sign_sequence.append(user_latest_sign[user_id])
        user_latest_sign[user_id] = sign_sequence[cnt]

    item_previous_sign_sequence = []
    item_latest_sign = defaultdict(lambda: 0)
    for cnt, item in enumerate(item_sequence):
        item_id = item2id[item]
        item_previous_sign_sequence.append(item_latest_sign[item_id])
        item_latest_sign[item_id] = sign_sequence[cnt]

    # for time-decay GCN
    timedifference_sequence_for_adj = (np.array(timestamp_sequence)/(3600*24)).astype(int)  # based on day
    

    if time_scaling:
        print("Scaling timestamps")

        # User polarity-aware timediffs
        user_pre_delta = (
            np.asarray(user_timedifference_sequence, dtype=np.float32) + 1.0
        )
        user_pos_delta = (
            np.asarray(user_pos_timedifference_sequence, dtype=np.float32) + 1.0
        )
        user_neg_delta = (
            np.asarray(user_neg_timedifference_sequence, dtype=np.float32) + 1.0
        )

        user_delta_mean = user_pre_delta.mean()
        user_delta_std = user_pre_delta.std()

        if user_delta_std < 1e-8:
            user_delta_std = 1.0

        user_timedifference_sequence = (
            user_pre_delta - user_delta_mean
        ) / user_delta_std

        user_pos_timedifference_sequence = (
            user_pos_delta - user_delta_mean
        ) / user_delta_std

        user_neg_timedifference_sequence = (
            user_neg_delta - user_delta_mean
        ) / user_delta_std

        item_timedifference_sequence = scale(
            np.asarray(item_timedifference_sequence) + 1
        )


    if is_kuairand:
        click_onehot = np.zeros(
            (len(rating_sequence), 2),
            dtype=np.float32
        )

        click_onehot[
            np.arange(len(rating_sequence)),
            rating_sequence.astype(int)
        ] = 1.0

        rating_sequence = click_onehot
    else:
        rating_onehot = np.zeros((len(rating_sequence), 5))
        rating_onehot[np.arange(len(rating_sequence)), (rating_sequence - 1).astype(int)] = 1
        rating_sequence = rating_onehot  # (num_interactions, 5)
    
    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_pos_timedifference_sequence,
    user_neg_timedifference_sequence, user_previous_itemid_sequence, item_previous_userid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, feature_sequence, rating_sequence, sign_sequence, \
        user_previous_sign_sequence, item_previous_sign_sequence, timedifference_sequence_for_adj]
    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def str2bool(value):
    """Converts string to boolean"""
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError("Boolean value expected")
    

def set_up_logger(args, sys_argv, now, log_path, checkpoint_root, best_model_root):
    runtime_id = '{}-{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute,  now.second, args.dataset)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    directory = log_path + '{}/{}/'.format(args.dataset, args.baseline)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    file_path = directory + '{}.log'.format(runtime_id)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)

    checkpoint_dir = checkpoint_root + runtime_id + '/'
    best_model_dir = best_model_root + runtime_id + '/'
    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root, exist_ok=True)
        logger.info('Create directory {}'.format(checkpoint_root))
    if not os.path.exists(best_model_root):
        os.makedirs(best_model_root, exist_ok=True)
        logger.info('Create directory'.format(best_model_root))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda epoch: (checkpoint_dir + 'checkpoint-epoch-{}.pth'.format(epoch))
    best_model_path = best_model_dir + 'best-model.pth'

    return logger, get_checkpoint_path, best_model_path


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2) 
        return allocated, reserved, peak
    return 0, 0, 0

def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()