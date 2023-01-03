import os
import json
from typing import List
import torch
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from data.base_dataset import BaseDataset
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


## copy from cpm-net
def random_mask(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return: Sn [alldata_len, view_num]
    """
    # print (f'==== generate random mask ====')
    one_rate = 1-missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum=2] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num)) # [samplenum, viewnum=2] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        # print (f'first ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        # print (f'second ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        # print (f'third ratio: {ratio}')
        error = abs(one_rate - ratio)
        
    return matrix



class IEMOCAPSIXMissDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.mask_rate = opt.mask_rate
        self.dataset = opt.dataset_mode.split('_')[0]
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAPSIX_config.json')))
        # load feature
        self.all_A = np.load(os.path.join(config['feature_root'], 'A', str(opt.cvNo), f'{set_name}.npy'), 'r')
        self.all_V = np.load(os.path.join(config['feature_root'], 'V', str(opt.cvNo), f'{set_name}.npy'), 'r')
        self.all_L = np.load(os.path.join(config['feature_root'], 'L', str(opt.cvNo), f'{set_name}.npy'), 'r')
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        self.int2name = np.load(int2name_path)
        # make missing index
        samplenum = len(self.label)
        self.maskmatrix = random_mask(3, samplenum, self.mask_rate) # [samplenum, view_num]

        self.manual_collate_fn = False

    def __getitem__(self, index):
        
        maskseq = self.maskmatrix[index] # (3, )
        assert np.sum(maskseq) >= 0.999999 # at least one view is not masked
        missing_index = torch.LongTensor(maskseq) # (3, ) [1,1,1]; maskrate=1=>[0,0,0]

        int2name = self.int2name[index]
        if self.dataset in ['cmumosi']:
            label = torch.tensor(self.label[index]).float()
        elif self.dataset in ['iemocapfour', 'iemocapsix']:
            label = torch.tensor(self.label[index]).long()
        A_feat = torch.tensor(self.all_A[index]).float()
        V_feat = torch.tensor(self.all_V[index]).float()
        L_feat = torch.tensor(self.all_L[index]).float()
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index
        }
    
    def __len__(self):
        return len(self.label)
