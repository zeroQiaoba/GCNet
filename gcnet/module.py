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


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type()) # [batch, seq_len]

        if self.att_type=='dot':
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # [batch, mem_dim, seqlen]
            x_ = self.transform(x).unsqueeze(1) # [batch, 1, mem_dim]
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # [batch, mem_dim, seq_len]
            M_ = M_ * mask_ # [batch, mem_dim, seqlen]
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1) # attention value: [batch, 1, seqlen]
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2) # [batch, 1, seqlen]
            alpha_masked = alpha_*mask.unsqueeze(1) # [batch, 1, seqlen]
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # [batch, 1, 1]
            alpha = alpha_masked/alpha_sum # normalized attention: [batch, 1, seqlen]
            # alpha = torch.where(alpha.isnan(), alpha_masked, alpha) 
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # [batch, 1, seqlen]

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # [batch, mem_dim]
        return attn_pool, alpha


# change [num_utterance, dim] => [seqlen, batch, dim]
def utterance_to_conversation(outputs, seq_lengths, umask, no_cuda):
    input_conversation_length = torch.tensor(seq_lengths) # [6, 24, 13, 9]
    start_zero = input_conversation_length.data.new(1).zero_() # [0]
    
    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths) # [int]
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0) # [0,  6, 30, 43]

    outputs = torch.stack([pad(outputs.narrow(0, s, l), max_len, no_cuda) # [seqlen, batch, dim]
                                for s, l in zip(start.data.tolist(),
                                input_conversation_length.data.tolist())], 0).transpose(0, 1)
    return outputs


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor
