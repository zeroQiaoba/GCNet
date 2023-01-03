import os
import glob
import shutil
import h5py
import json
import numpy as np
from numpy.lib.function_base import extract
from tqdm import tqdm
import pickle as pkl

from preprocess.tools.denseface_extractor import DensefaceExtractor
from preprocess.tools.bert_extractor import BertExtractor
from preprocess.MSP.make_comparE import ComParEExtractor

def load_A(config, save_path):
    if os.path.exists(save_path):
        print('All comparE feat found in {}'.format(save_path))
        all_feat = pkl.load(open(save_path, 'rb'))
    else:
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
            all_feat[utt_id] = feat
        pkl.dump(all_feat, open(save_path, 'wb'))
    return all_feat

def make_all_comparE(config):
    all_feat_path = os.path.join(config['data_root'], 'Audio', 'all_comparE.pkl')
    feat = load_A(config, all_feat_path)
    
    # process feat and record timestamp
    feat_save_path = os.path.join(config['feature_root'], 'aligned', "A", "raw_comparE.h5")
    h5f = h5py.File(feat_save_path, 'w')
    for utt_id in tqdm(feat.keys()):
        utt_feat = feat[utt_id][()]
        start = np.array([0.1 * x for x in range(len(utt_feat))])
        end = start + 0.1
        utt_group = h5f.create_group(utt_id)
        utt_group['feat'] = feat[utt_id][()]
        utt_group['start'] = start
        utt_group['end'] = end


def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label


def get_all_utt_id(config):
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = trn_int2name.tolist()
    val_int2name = val_int2name.tolist()
    tst_int2name = tst_int2name.tolist()
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    return all_utt_ids

def make_all_denseface(config):
    face_root = os.path.join(config['data_root'], 'Face', '{}')
    extractor = DensefaceExtractor(device=0)
    all_utt_ids = get_all_utt_id(config)
    feat_save_path = os.path.join(config['feature_root'], 'aligned', "V", "raw_denseface.h5")
    h5f = h5py.File(feat_save_path, 'w')
    for utt_id in tqdm(all_utt_ids):
        face_dir = face_root.format(utt_id)
        utt_face_pics = sorted(glob.glob(os.path.join(face_dir, '*.jpg')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        utt_feats = []
        utt_start = []
        utt_end = []
        for pic_path in utt_face_pics:
            frame_id = int(pic_path.split('/')[-1].split('.')[0])
            feat, _ = extractor(pic_path)
            timestamp = (frame_id - 1) / 30
            utt_feats.append(feat)
            utt_start.append(timestamp)
            utt_end.append(timestamp + 0.1)
        if len(utt_feats) != 0:
            utt_feats = np.concatenate(utt_feats, axis=0)
            utt_start = np.array(utt_start)
            utt_end = np.array(utt_end)
        else:
            utt_feats = np.zeros([1, 342])
            utt_start = np.array([-1])
            utt_end = np.array([-1])
        utt_group = h5f.create_group(utt_id)
        utt_group['feat'] = utt_feats
        utt_group['start'] = utt_start
        utt_group['end'] = utt_end


def calc_real_time(frm):
    frm = int(frm)
    return frm / 100


# def read_align_file(file):
#     lines = open(file).readlines()
#     lines = [x.strip() for x in lines]
#     phone_st = lines.index('item [2]:')
#     lines = lines[:phone_st]
#     ans = []
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         if line.startswith('intervals [') and line.endswith(':'):
#             start = float(lines[i+1].split('=')[1].strip())
#             end = float(lines[i+2].split('=')[1].strip())
#             word = lines[i+3].split('=')[1].strip().replace('"', '').strip()
            
        
#             one_record = {
#                 'start_time':start,
#                 'end_time':end,
#                 'word': word,
#             }
#             if len(word) != 0 and word.lower() != 'inaudible':
#                 ans.append(one_record)
#                 i += 3
        
#         i += 1
#     return ans

def read_align_file(file):
    ans = []
    data = json.load(open(file))
    data = data['words']
    for word_info in data:
        if word_info['word'] != '{p}' and len(word_info['word']) > 0:
            one_record = {
                'start_time':word_info['start'],
                'end_time':word_info['end'],
                'word': word_info['word'],
            }
            ans.append(one_record)
    return ans


def make_all_bert(config):
    # from debug import show_wdseg, show_sentence
    extractor = BertExtractor(cuda=True, cuda_num=0)
    word_info_dir = os.path.join(config['feature_root'], 'aligned/word_aligned_info')
    all_utt_ids = get_all_utt_id(config)
    feat_save_path = os.path.join(config['feature_root'], 'aligned', "L", "raw_bert.h5")
    h5f = h5py.File(feat_save_path, 'w')
    count = 0
    for utt_id in tqdm(all_utt_ids):
        word_info_path = os.path.join(word_info_dir, utt_id + ".json")
        count += 1
        word_infos = read_align_file(word_info_path)
        word_lst = [x["word"] for x in word_infos]
        # print("WORDS:", word_lst)
        token_ids, word_idxs = extractor.tokenize(word_lst)
        utt_start = [word_infos[i]['start_time'] for i in word_idxs]
        utt_end = [word_infos[i]['end_time'] for i in word_idxs]
        utt_feats, _ = extractor.get_embd(token_ids)
        utt_feats = utt_feats.squeeze(0).cpu().numpy()[1:-1, :]
        assert utt_feats.shape[0] == len(utt_end)
        utt_group = h5f.create_group(utt_id)
        utt_group['feat'] = utt_feats
        utt_group['start'] = utt_start
        utt_group['end'] = utt_end
    print(count)
        # show_wdseg(utt_id)
        # show_sentence(utt_id)
        # input()

def make_aligned_data(config):
    raw_A_path = os.path.join(config['feature_root'], 'aligned', "A", "raw_comparE.h5")
    raw_V_path = os.path.join(config['feature_root'], 'aligned', "V", "raw_denseface.h5")
    raw_L_path = os.path.join(config['feature_root'], 'aligned', "L", "raw_bert.h5")
    raw_A = h5py.File(raw_A_path, 'r')
    raw_V = h5py.File(raw_V_path, 'r')
    raw_L = h5py.File(raw_L_path, 'r')
    all_utt_ids = get_all_utt_id(config)
    aligned_A_path = os.path.join(config['feature_root'], 'aligned', "A", "aligned_comparE.h5")
    aligned_V_path = os.path.join(config['feature_root'], 'aligned', "V", "aligned_denseface.h5")
    aligned_L_path = os.path.join(config['feature_root'], 'aligned', "L", "aligned_bert.h5")
    aligned_A_h5f = h5py.File(aligned_A_path, 'w')
    aligned_V_h5f = h5py.File(aligned_V_path, 'w')
    aligned_L_h5f = h5py.File(aligned_L_path, 'w')

    for utt_id in tqdm(all_utt_ids):
        if utt_id == 'Ses03M_impro03_M001': # 这个语句缺少对齐信息文件
            continue
        utt_A_feat, utt_A_start, utt_A_end = \
            raw_A[utt_id]['feat'][()], raw_A[utt_id]['start'][()], raw_A[utt_id]['end'][()]
        utt_V_feat, utt_V_start, utt_V_end = \
            raw_V[utt_id]['feat'][()], raw_V[utt_id]['start'][()], raw_V[utt_id]['end'][()]
        utt_L_feat, utt_L_start, utt_L_end = \
            raw_L[utt_id]['feat'][()], raw_L[utt_id]['start'][()], raw_L[utt_id]['end'][()]
        
        utt_aligned_A, utt_aligned_V = [], []
        for word_start, word_end in zip(utt_L_start, utt_L_end):
            word_aligned_a = calc_word_aligned(word_start, word_end, utt_A_feat, utt_A_start, utt_A_end, default_dim=130)
            word_aligned_v = calc_word_aligned(word_start, word_end, utt_V_feat, utt_V_start, utt_V_end, default_dim=342)
            utt_aligned_A.append(word_aligned_a)
            utt_aligned_V.append(word_aligned_v)
        
        utt_aligned_A = np.array(utt_aligned_A)
        utt_aligned_V = np.array(utt_aligned_V)
        assert(len(utt_aligned_A) == len(utt_aligned_V) == len(utt_L_feat))
        # print(f'A:{utt_aligned_A.shape} V:{utt_aligned_V.shape} L:{utt_L_feat.shape}')
        aligned_A_h5f[utt_id] = utt_aligned_A
        aligned_V_h5f[utt_id] = utt_aligned_V
        aligned_L_h5f[utt_id] = utt_L_feat

def calc_word_aligned(word_start, word_end, frame_feats, frame_start, frame_end, default_dim=342):
    _frame_set = []
    assert word_end > word_start
    for feat, start, end in zip(frame_feats, frame_start, frame_end):
        if start == end == -1 and np.sum(frame_feats) == 0:
            break
        assert end > start, f'{start}, {end}, {frame_feats}'
        if start > word_end or end < word_start:
            continue
        else:
            _frame_set.append(feat)
    if len(_frame_set) > 0:
        _frame_set = np.array(_frame_set)
    else:
        _frame_set = np.zeros([1, default_dim])
    return np.mean(_frame_set, axis=0)

def normlize_on_trn(config, input_file, output_file):
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 13):
        trn_int2name, _ = get_trn_val_tst(config['target_root'], cv, 'trn')
        trn_int2name = trn_int2name.tolist()
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name]
        all_feat = np.concatenate(all_feat, axis=0)
        mean_f = np.mean(all_feat, axis=0)
        std_f = np.std(all_feat, axis=0)
        std_f[std_f == 0.0] = 1.0
        cv_group = h5f.create_group(str(cv))
        cv_group['mean'] = mean_f
        cv_group['std'] = std_f
        print(cv)
        print("mean:", np.sum(mean_f))
        print("std:", np.sum(std_f))



if __name__ == '__main__':
    # load config

    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../', 'data/config', 'MSP_config.json')
    config = json.load(open(config_path))

    # all_feat_path = os.path.join(config['data_root'], 'Audio', 'all_comparE.pkl')
    # feat = load_A(config, all_feat_path)

    save_dir = os.path.join(config['feature_root'], 'aligned')
    for modality in ['A', 'V', 'L']:
        modality_dir = os.path.join(save_dir, modality)
        if not os.path.exists(modality_dir):
            os.makedirs(modality_dir)

    # make raw feat record with timestamp
    # make_all_comparE(config)
    # make_all_denseface(config)
    # make_all_bert(config)

    # make_aligned_data
    # make_aligned_data(config)

    # normalize A feat
    normlize_on_trn(config,
        os.path.join(config['feature_root'], 'aligned', 'A', 'aligned_comparE.h5'), 
        os.path.join(config['feature_root'], 'aligned', 'A', 'aligned_comparE_mean_std.h5')
    )
