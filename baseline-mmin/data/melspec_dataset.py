import os
import json
import torch
import numpy as np
import h5py
import PIL
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset


class MelspecDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--spec_aug', action='store_true', help='whether to do specaug')
        parser.add_argument('--time_mask', type=float, default=0.1, help='specaug parameter time mask')
        parser.add_argument('--freq_mask', type=float, default=0.1, help='specaug parameter freq mask')
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
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAP_config.json')))
        
        # load feature
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'A', 'melspec.h5'), 'r')

        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(int2name_path)

        # loading setting
        self.spec_aug = opt.spec_aug
        self.time_mask = opt.time_mask
        self.freq_mask = opt.freq_mask
        
        self.manual_collate_fn = False

    def get_transform(self):
        if self.set_name == 'trn':
            _transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            _transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        return _transform

    def process_melspec(self, melspec):
        if not hasattr(self, "transform"):
            self.transform = self.get_transform()
        
        image = melspec
        time_dim, base_dim = image.shape[1], image.shape[0]
        crop = np.random.randint(0, time_dim - base_dim)
        image = image[:, crop:crop + base_dim, ...]

        if self.set_name == 'trn' and self.spec_aug:
            freq_mask_begin = int(np.random.uniform(0, 1 - self.freq_mask) * base_dim)
            image[freq_mask_begin:freq_mask_begin + int(self.freq_mask * base_dim), ...] = 0
            time_mask_begin = int(np.random.uniform(0, 1 - self.time_mask) * base_dim)
            image[:, time_mask_begin:time_mask_begin + int(self.time_mask * base_dim), ...] = 0

        image = PIL.Image.fromarray(image[...,0], mode='L')
        image = self.transform(image).div_(255)
        return image.float()

    def __getitem__(self, index):
        int2name = self.int2name[index][0].decode()
        label = torch.tensor(self.label[index])
        # process A_feat
        A_feat = self.all_A[int2name]
        A_feat = self.process_melspec(A_feat)
        return {
            'A_feat': A_feat, 
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        spec_aug = True
        time_mask = 0.1
        freq_mask = 0.1
    
    opt = test()
    a = MelspecDataset(opt, set_name='trn')
    data = next(iter(a))
    for k, v in data.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    