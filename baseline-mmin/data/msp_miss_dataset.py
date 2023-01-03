import os
import json
from typing import List
import torch
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class MSPMissDataset(BaseDataset):
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
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', 'MSP_config.json')))
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
        if set_name != 'trn':           # val && tst
            self.missing_index = torch.tensor([
                [1,0,0], # AZZ
                [0,1,0], # ZVZ
                [0,0,1], # ZZL
                [1,1,0], # AVZ
                [1,0,1], # AZL
                [0,1,1], # ZVL
                # [1,1,1]  # AVL
            ] * len(self.label)).long()
            self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'] * len(self.label)
        else:                           # trn
            self.missing_index = [
                [1,0,0], # AZZ
                [0,1,0], # ZVZ
                [0,0,1], # ZZL
                [1,1,0], # AVZ
                [1,0,1], # AZL
                [0,1,1], # ZVL
                # [1,1,1]  # AVL
            ]
            self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']
        self.manual_collate_fn = False

    def __getitem__(self, index):
        if self.set_name != 'trn':
            feat_idx = index // 6         # totally 6 missing types
            missing_index = self.missing_index[index]         
            miss_type = self.miss_type[index]
        else:
            feat_idx = index
            missing_index = torch.tensor(random.choice(self.missing_index)).long()
            miss_type = random.choice(self.miss_type)
        
        int2name = self.int2name[feat_idx]
        label = torch.tensor(self.label[feat_idx])
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[feat_idx]).float()
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[feat_idx]).float()
        # proveee L_feat
        L_feat = torch.from_numpy(self.all_L[feat_idx]).float()
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        } if self.set_name == 'trn' else{
            'A_feat': A_feat * missing_index[0], 
            'V_feat': V_feat * missing_index[1],
            'L_feat': L_feat * missing_index[2],
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        }
    
    def __len__(self):
        return len(self.missing_index) if self.set_name != 'trn' else len(self.label)
    
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

if __name__ == '__main__':
    class test:
        cvNo = 1
    
    opt = test()
    print('Reading from dataset:')
    a = MSPMissDataset(opt, set_name='trn')
    data = next(iter(a))
    for k, v in data.items():
        if k not in ['int2name', 'label', 'miss_type']:
            print(k, v.shape)
        else:
            print(k, v)
    