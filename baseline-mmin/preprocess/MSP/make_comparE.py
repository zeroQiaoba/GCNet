import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from torch.nn.functional import normalize
from tqdm import tqdm


class ComParEExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, opensmile_tool_dir=None, downsample=10, tmp_dir='.tmp', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            opensmile_tool_dir = '/root/opensmile-2.3.0/'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, wav):
        basename = os.path.basename(wav).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        os.system(cmd.format(self.opensmile_tool_dir, wav, save_path))
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_data = df.iloc[:, 2:]
        if len(wav_data) > self.downsample:
            wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
            if self.no_tmp:
                os.remove(save_path) 
        else:
            wav_data = None
            self.print(f'Error in {wav}, no feature extracted')

        return wav_data


def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def padding_to_fixlen(feat, max_len):
    assert feat.ndim == 2
    if feat.shape[0] >= max_len:
        feat = feat[:max_len]
    else:
        feat = np.concatenate([feat, \
            np.zeros((max_len-feat.shape[0], feat.shape[1]))], axis=0)
    return feat

def make_all_comparE(config):
    max_len = 50
    extractor = ComParEExtractor()
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = trn_int2name.tolist()
    val_int2name = val_int2name.tolist()
    tst_int2name = tst_int2name.tolist()
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_feat = {}
    for utt_id in tqdm(all_utt_ids): # MSP-IMPROV-S01A-F01-S-FM01
        ses_id = int(utt_id.split('-')[3][-1])
        dialog_id = utt_id.split('-')[2]
        wav_path = os.path.join(config['data_root'], 'Audio', f'session{ses_id}', dialog_id, 'S', f'{utt_id}.wav')
        feat = extractor(wav_path)
        all_feat[utt_id] = padding_to_fixlen(feat, max_len)
    
    for cv in range(1, config['total_cv']+1):
        save_dir = os.path.join(config['feature_root'], 'A', str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for set_name in ['trn', 'val', 'tst']:
            int2name, _ = get_trn_val_tst(config['target_root'], cv, set_name)
            cv_feats = []
            for utt_id in int2name:
                cv_feats.append(all_feat[utt_id])
            cv_feats = np.array(cv_feats)
            cv_feats = normalize(cv_feats)
            save_path = os.path.join(save_dir, set_name + '.npy')
            print(f'fold:{cv} {set_name} {cv_feats.shape}')
            np.save(save_path, cv_feats)

def normalize(feats):
    _feats = feats.reshape(-1, feats.shape[2])
    mean = np.mean(_feats, axis=0)
    std = np.std(_feats, axis=0)
    std[std == 0.0] = 1.0
    ret = (feats-mean) / (std)
    return ret
    
if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../', 'data/config', 'MSP_config.json')
    config = json.load(open(config_path))
    make_all_comparE(config)