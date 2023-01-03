import os
import time
import glob
import pickle
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv


def edge_perms(l, window_past, window_future):
    """
    Target:
        Method to construct the edges considering the past and future window.
    
    Input: 
        l: seq length
        window_past, window_future: context lengths

    Output:
        all_perms: all connected edges
    """
    all_perms = set()
    array = np.arange(l)
    for j in range(l): # j: start index
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)



## accurate graph building process [single relation graph]
def batch_graphify(features, qmask, lengths, n_speakers, window_past, window_future, graph_type, no_cuda):
    """
    Target: prepare the data format required for the GCN network.
    Different batches have no edge connection.

    qmask: save speaker items [Batch, Time] !!!! [tensor]
    features: [Time, Batch, ?, Feat] => for node initialization [tensor]
    lengths: each conversation has its own lens [int]
    window_past, window_future: context lens [int]

    'one_nms', 'one_ms', 'two_nms', 'two_ms':
    one/two means one speaker per time; or two speakers per time
    ms/nms means modality-specific and non modality-specific
    """
    ## define edge_type_mapping
    order_types = ['past', 'now', 'future']
    assert n_speakers <= 2, 'Note: n_speakers mush <= 2'
    if n_speakers == 1: speaker_types = ['00']
    if n_speakers == 2: speaker_types = ['00', '01', '10', '11']
    
    ## only for single relation graph
    assert graph_type in ['temporal', 'speaker'] 
    merge_types = set()
    if graph_type == 'temporal':
        for ii in range(len(order_types)):
            merge_types.add(f'{order_types[ii]}')
    elif graph_type == 'speaker':
        for ii in range(len(speaker_types)):
            merge_types.add(f'{speaker_types[ii]}')
    
    edge_type_mapping = {}
    for ii, item in enumerate(merge_types):
        edge_type_mapping[item] = ii

    ## qmask to cup()
    qmask = qmask.cpu().data.numpy().astype(int)

    ## build graph
    node_features = []
    edge_index, edge_type = [], []
    length_sum = 0 # for unique node index
    batch_size = features.size(1)
    for j in range(batch_size):
        # gain node_features
        node_feature = features[:lengths[j], j, :, :] # [Time, Batch, ?, Feat] -> [Time, ?, Feat]
        node_feature = torch.reshape(node_feature, (-1, node_feature.size(-1))) # [Time*?, Feat]
        node_features.append(node_feature) # [Time*?, Feat]
        
        # make sure different conversations have no connection
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j] # add node number [no repeated nodes]
        
        ## change perms1 and perms2
        for item1, item2 in zip(perms1, perms2):

            # gain edge_index [actual edge]
            edge_index.append([item2[0], item2[1]])
            
            # gain edge_type
            (jj, ii) = (item1[0], item1[1])

            ## item1: gain time order
            jj_time = int(jj)
            ii_time = int(ii)
            if ii_time > jj_time:
                order_type = 'past'
            elif ii_time == jj_time:
                order_type = 'now'
            else:
                order_type = 'future'

            ## item2 gain speaker order: [host:0, guest:1]
            ## for one speaker, only has 'host2host'
            jj_speaker = qmask[j, jj_time]
            ii_speaker = qmask[j, ii_time]
            speaker_type = f'{ii_speaker}{jj_speaker}'

            ## merge [item1, item2, item3]
            if graph_type == 'speaker':  edge_type_name = f'{speaker_type}'
            if graph_type == 'temporal': edge_type_name = f'{order_type}'
            edge_type.append(edge_type_mapping[edge_type_name])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.tensor(edge_index).transpose(0, 1)
    edge_type = torch.tensor(edge_type)

    #if torch.cuda.is_available():
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()
    
    return node_features, edge_index, edge_type, edge_type_mapping