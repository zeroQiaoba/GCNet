import os
import h5py
import json
import numpy as np
from tqdm import tqdm

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label


def migrate_V(config):
    migrate_root = os.path.join('/data3/lrc/Iemocap_feature/cv_level/feature/denseface/', str(1))
    src_v_trn = np.load(os.path.join(migrate_root, 'trn.npy'))
    src_v_val = np.load(os.path.join(migrate_root, 'val.npy'))
    src_v_tst = np.load(os.path.join(migrate_root, 'tst.npy'))
    src_v_feat = np.concatenate([src_v_trn, src_v_val, src_v_tst], axis=0)
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'V', 'denseface.h5'), 'w')
    for utt_id, v_feat in tqdm(zip(all_utt_ids, src_v_feat), total=len(all_utt_ids)):
        all_h5f[utt_id] = v_feat


def migrate_L(config):
    migrate_root = os.path.join('/data3/lrc/Iemocap_feature/cv_level/feature/text/', str(1))
    src_l_trn = np.load(os.path.join(migrate_root, 'trn.npy'))
    src_l_val = np.load(os.path.join(migrate_root, 'val.npy'))
    src_l_tst = np.load(os.path.join(migrate_root, 'tst.npy'))
    src_l_feat = np.concatenate([src_l_trn, src_l_val, src_l_tst], axis=0)
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'L', 'bert_large.h5'), 'w')
    for utt_id, l_feat in tqdm(zip(all_utt_ids, src_l_feat), total=len(all_utt_ids)):
        all_h5f[utt_id] = l_feat


if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    # migrate_V(config)
    migrate_L(config)
