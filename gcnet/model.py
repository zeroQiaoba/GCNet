import os
import time
import glob
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv

from module import *
from graph import batch_graphify


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_relations, time_attn, hidden_size=64, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        self.no_cuda = no_cuda 
        self.time_attn = time_attn
        self.hidden_size = hidden_size

        ## graph modeling
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
        self.conv2 = GraphConv(hidden_size, hidden_size)

        ## nodal attention
        D_h = num_features+hidden_size
        self.grufusion = nn.LSTM(input_size=D_h, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout)

        ## sequence attention
        self.matchatt = MatchingAttention(2*D_h, 2*D_h, att_type='general2')
        self.linear = nn.Linear(2*D_h, D_h)


    def forward(self, features, edge_index, edge_type, seq_lengths, umask):
        '''
        features: input node features: [num_nodes, in_channels]
        edge_index: [2, edge_num]
        edge_type: [edge_num]
        '''

        ## graph model: graph => outputs
        out = self.conv1(features, edge_index, edge_type) # [num_features -> hidden_size]
        out = self.conv2(out, edge_index) # [hidden_size -> hidden_size]
        outputs = torch.cat([features, out], dim=-1) # [num_nodes, num_features(16)+hidden_size(8)]

        ## change utterance to conversation: (outputs->outputs)
        outputs = outputs.reshape(-1, outputs.size(1)) # [num_utterance, dim]
        outputs = utterance_to_conversation(outputs, seq_lengths, umask, self.no_cuda) # [seqlen, batch, dim]
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1) # [seqlen, batch, ?, dim]

        ## outputs -> outputs:
        seqlen = outputs.size(0)
        batch = outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1)) # [seqlen, batch, dim]
        outputs = self.grufusion(outputs)[0] # [seqlen, batch, dim]

        ## outputs -> hidden:
        ## sequence attention => [seqlen, batch, d_h]
        if self.time_attn:
            alpha = []
            att_emotions = []
            for t in outputs: # [bacth, dim]
                # att_em: [batch, mem_dim] # alpha_: [batch, 1, seqlen]
                att_em, alpha_ = self.matchatt(outputs, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0)) # [1, batch, mem_dim]
                alpha.append(alpha_[:,0,:]) # [batch, seqlen]
            att_emotions = torch.cat(att_emotions, dim=0) # [seqlen, batch, mem_dim]
            hidden = F.relu(self.linear(att_emotions)) # [seqlen, batch, D_h]
        else:
            alpha = []
            hidden = F.relu(self.linear(outputs)) # [seqlen, batch, D_h]

        return hidden # [seqlen, batch, D_h]

        
'''
base_model: LSTM or GRU
adim, tdim, vdim: input feature dim
D_e: hidder feature dimensions of base_model is 2*D_e
D_g, D_p, D_h, D_a, graph_hidden_size
'''
class GraphModel(nn.Module):

    def __init__(self, base_model, adim, tdim, vdim, D_e, graph_hidden_size, n_speakers, window_past, window_future,
                 n_classes ,dropout=0.5, time_attn=True, no_cuda=False):
        
        super(GraphModel, self).__init__()

        self.no_cuda = no_cuda
        self.base_model = base_model

        # The base model is the sequential context encoder.
        # Change input features => 2*D_e
        self.lstm = nn.LSTM(input_size=adim+tdim+vdim, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru = nn.GRU(input_size=adim+tdim+vdim, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
       
        ## Defination for graph model
        ## [modality_type=3(AVT); time_order=3(past, now, future)]
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future
        self.time_attn = time_attn

        ## gain graph models for 'temporal' and 'speaker'
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout, self.no_cuda)
        n_relations = n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout, self.no_cuda)

        ## classification and reconstruction
        D_h = 2*D_e + graph_hidden_size
        self.smax_fc  = nn.Linear(D_h, n_classes)
        self.linear_rec = nn.Linear(D_h, adim+tdim+vdim)

    def forward(self, inputfeats, qmask, umask, seq_lengths):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        """

        ## sequence modeling
        ## inputfeats -> outputs [seqlen, batch, ?, dim]
        if self.base_model == 'LSTM':
            outputs, _ = self.lstm(inputfeats[0])
            outputs = outputs.unsqueeze(2)
        elif self.base_model == 'GRU':
            outputs, _ = self.gru(U[0])
            outputs = outputs.unsqueeze(2)

        ## add graph model
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'temporal', self.no_cuda)
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'speaker', self.no_cuda)
        assert len(edge_type_mapping) == self.n_speakers ** 2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, seq_lengths, umask)
        hidden = hidden1 + hidden2

        ## for classification
        log_prob = self.smax_fc(hidden) # [seqlen, batch, n_classes]

        ## for reconstruction
        rec_outputs = [self.linear_rec(hidden)]

        return log_prob, rec_outputs, hidden

