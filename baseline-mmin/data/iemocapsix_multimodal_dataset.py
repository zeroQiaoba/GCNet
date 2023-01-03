import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class IEMOCAPSIXMultimodalDataset(BaseDataset):
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
        self.set_name = set_name
        self.dataset = opt.dataset_mode.split('_')[0]
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAPSIX_config.json')))
        self.norm_method = opt.norm_method

        # load feature
        self.all_A = np.load(os.path.join(config['feature_root'], 'A', str(opt.cvNo), f'{set_name}.npy'), 'r')
        self.all_V = np.load(os.path.join(config['feature_root'], 'V', str(opt.cvNo), f'{set_name}.npy'), 'r')
        self.all_L = np.load(os.path.join(config['feature_root'], 'L', str(opt.cvNo), f'{set_name}.npy'), 'r')

        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        self.int2name = np.load(int2name_path)
        self.manual_collate_fn = False ## for utterance level features

    def __getitem__(self, index):
        int2name = self.int2name[index]
        if self.dataset in ['cmumosi']:
            label = torch.tensor(self.label[index]).float()
        elif self.dataset in ['iemocapfour', 'iemocapsix']:
            label = torch.tensor(self.label[index]).long()
        # process A_feat
        A_feat = torch.FloatTensor(self.all_A[index])
        # process V_feat 
        V_feat = torch.FloatTensor(self.all_V[index])
        # proveee L_feat
        L_feat = torch.FloatTensor(self.all_L[index])
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)
    
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }
