import os
import json
import numpy as np
from tqdm import tqdm
import re
import string

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

def align_script(wav, config, out):
    _cmd = 'python /data6/p2fa-vislab/align.py {} {} {} '.format(wav, config, out) # >/dev/null 2>&1
    os.system(_cmd)

def clean(text):
    punc = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    text = re.sub(r"[%s]+" %punc, " ",text)
    text.replace('  ', ' ')
    return text

def make_aligned_info(config):
    save_dir = os.path.join(config['feature_root'], 'aligned', 'word_aligned_info')
    tmp_dir = os.path.join(config['feature_root'], 'aligned', 'tmp')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    transcript_dir = os.path.join(config['data_root'], 'All_human_transcriptions')
    all_utt_ids = get_all_utt_id(config)
    for utt_id in tqdm(all_utt_ids):
        align_save_path = os.path.join(save_dir, utt_id + '.json')
        if os.path.exists(align_save_path):
            continue
        wav_dir = os.path.join(config['feature_root'], 'audio_11025')
        wav_path = os.path.join(wav_dir, '{}.wav'.format(utt_id))
        transcript_path = os.path.join(transcript_dir, utt_id + '.txt')
        transcript = open(transcript_path).read().strip()
        transcript = clean(transcript)
        print('"' + transcript + '"')
        tmp_path = os.path.join(tmp_dir, utt_id + '.json')
        tmp_data = [{
            "speaker": "Steve",
            "line": transcript,
        }]
        json.dump(tmp_data, open(tmp_path, 'w'))
        align_script(wav_path, tmp_path, align_save_path)

def convert_sr(config):
    sampled_audio_dir = os.path.join(config['feature_root'], 'audio_11025')
    if not os.path.exists(sampled_audio_dir):
        os.mkdir(sampled_audio_dir)
    all_utt_ids = get_all_utt_id(config)
    for utt_id in tqdm(all_utt_ids):
        ses_id = int(utt_id.split('-')[3][-1])
        dialog_id = utt_id.split('-')[2]
        wav_path = os.path.join(config['data_root'], 'Audio', \
            'session{}'.format(ses_id), dialog_id, 'S', '{}.wav'.format(utt_id))
        cmd = 'sox {} -r 11025 {}'
        new_audio_path = os.path.join(sampled_audio_dir, utt_id + '.wav')
        os.system(cmd.format(wav_path, new_audio_path))

if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../', 'data/config', 'MSP_config.json')
    config = json.load(open(config_path))
    make_aligned_info(config)
    # convert_sr(config)