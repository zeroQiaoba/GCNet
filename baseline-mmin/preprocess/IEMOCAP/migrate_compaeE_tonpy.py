import os
import json
import h5py
import numpy as np

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

def migrate_comparE_to_npy(config):
    max_len = 60
    feat_path = os.path.join(config['feature_root'], 'A', 'comparE.h5')
    mean_std_path = os.path.join(config['feature_root'], 'A', 'comparE_mean_std.h5')
    feat_h5f = h5py.File(feat_path, 'r')
    mean_std = h5py.File(mean_std_path, 'r')
    for cv in range(1, 11):
        save_dir = f'/data3/lrc/Iemocap_feature/cv_level/feature/comparE/{cv}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        mean = mean_std[str(cv)]['mean'][()]
        std = mean_std[str(cv)]['std'][()]
        for part in ['trn', 'val', 'tst']:
            part_feat = []
            int2name, _ = get_trn_val_tst(config['target_root'], cv, part)
            int2name = [x[0].decode() for x in int2name]
            for utt_id in int2name:
                feat = feat_h5f[utt_id][()]
                feat = (feat-mean)/std
                feat = padding_to_fixlen(feat, max_len)
                part_feat.append(feat)
            part_feat = np.array(part_feat)
            print(f"cv: {cv} {part} {part_feat.shape}")
            save_path = os.path.join(save_dir, f"{part}.npy")
            np.save(save_path, part_feat)
    
if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    migrate_comparE_to_npy(config)


