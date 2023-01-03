import os
import h5py
import json

def statis_comparE(config):
    path = os.path.join(config['feature_root'], 'A', 'comparE.h5')
    h5f = h5py.File(path, 'r')
    lengths = []
    for utt_id in h5f.keys():
        lengths.append(h5f[utt_id][()].shape[0])
    lengths = sorted(lengths)
    print('MIN:', min(lengths))
    print('MAX:', max(lengths))
    print('MEAN: {:.2f}'.format(sum(lengths) / len(lengths)))
    print('50%:', lengths[len(lengths)//2])
    print('75%:', lengths[int(len(lengths)*0.75)])
    print('90%:', lengths[int(len(lengths)*0.9)])

if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    statis_comparE(config)