import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class MultimodalDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
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
        pwd = os.path.abspath(__file__) # gain file folder
        pwd = os.path.dirname(pwd) # '/data5/lianzheng/MMIN-master/data'
        '''
        config = {'data_root': '/data5/lianzheng/MMIN-master/IEMOCAP_full_release/', 
        'target_root': '/data5/lianzheng/MMIN-master/IEMOCAP_features_2021/target', 
        'total_cv': 10, 
        'feature_root': '/data5/lianzheng/MMIN-master/IEMOCAP_features_2021'}
        '''
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAP_config.json'))) # load json file
        self.norm_method = opt.norm_method # 'trn'

        # load feature
        self.A_type = opt.A_type
        # self.all_A: maybe map 5531 sample feats, each sample is frame-level features [17, 130]
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'A', f'{self.A_type}.h5'), 'r')
        if self.A_type == 'comparE':
            self.mean_std = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE_mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float() # [1, 130]
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float() # [1, 130]
        self.V_type = opt.V_type
        self.all_V = h5py.File(os.path.join(config['feature_root'], 'V', f'{self.V_type}.h5'), 'r')
        self.L_type = opt.L_type
        self.all_L = h5py.File(os.path.join(config['feature_root'], 'L', f'{self.L_type}.h5'), 'r')
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path) # [4500, 4]
        self.label = np.argmax(self.label, axis=1) # [4500, ]
        self.int2name = np.load(int2name_path) # [4500, 1] names
        # in aligned dataset, you should move out Ses03M_impro03_M001
        # if 'Ses03M_impro03_M001' in self.int2name:
        #     idx = self.int2name.index('Ses03M_impro03_M001')
        #     self.int2name.pop(idx)
        #     self.label = np.delete(self.label, idx, axis=0)
        self.manual_collate_fn = True

    def __getitem__(self, index):
        int2name = self.int2name[index][0].decode() # str
        label = torch.tensor(self.label[index]) # int
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == 'comparE':
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # process L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)
    
    ## utterance-level normalization
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    ## train dataset-level normalization
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch] # [batch * [?, 130]]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long() # [?1, ?2, ..., ?N] => frame lens
        A = pad_sequence(A, batch_first=True, padding_value=0) # [batch, ?max, 130]
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch]) # [batch, 1]
        int2name = [sample['int2name'] for sample in batch]         # [batch, 1]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert_large"
        norm_method = 'trn'

    
    opt = test()
    print('Reading from dataset:')
    a = MultimodalDataset(opt, set_name='trn')
    data = next(iter(a))
    for k, v in data.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    print('Reading from dataloader:')
    x = [a[100], a[34], a[890]]
    print('each one:')
    for i, _x in enumerate(x):
        print(i, ':')
        for k, v in _x.items():
            if k not in ['int2name', 'label']:
                print(k, v.shape)
            else:
                print(k, v)
    print('packed output')
    x = a.collate_fn(x)
    for k, v in x.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    