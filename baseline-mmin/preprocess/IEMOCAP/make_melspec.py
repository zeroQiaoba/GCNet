import os
import json
import numpy as np
import h5py
from tqdm import tqdm
from preprocess.melspec_extractor import MelSpecExtractor

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def extract_all_melspec(config):
    extractor = MelSpecExtractor()
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'A', 'melspec.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        ses_id = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        wav_path = os.path.join(config['data_root'], f'Session{ses_id}', 'sentences', 'wav', f'{dialog_id}', f'{utt_id}.wav')
        melspec = extractor.extract(wav_path)
        all_h5f[utt_id] = melspec

if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    extract_all_melspec(config)
