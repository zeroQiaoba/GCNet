# *_*coding:utf-8 *_*
"""
wav2vec: https://arxiv.org/abs/1904.05862
official github repo: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
"""
import time
import os
import glob
import numpy as np
import pandas as pd
import torch
from fairseq.models.wav2vec import Wav2VecModel # Note: use fairseq version of 0.10.1, error occurred when using the newest officical script and version of 0.10.2 (pip install fairseq==0.10.1)
from util import write_feature_to_csv, write_feature_to_npy
import argparse
import soundfile as sf

# import config
import sys
sys.path.append('../../')
import config

# save_dir/dir_name
def extract(audio_files, feature_level, model, save_dir, label_interval, dir_name=None, overwrite=False, gpu=None):
    start_time = time.time()
    device = torch.device(f'cuda:{gpu}')

    # out dir
    if dir_name is None:
        dir_name = 'wav2vec-large'
    out_dir_z = os.path.join(save_dir, f'{dir_name}-z-{feature_level[:3]}') # features output by feature encoder
    if not os.path.exists(out_dir_z):
        os.makedirs(out_dir_z)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    out_dir_c = os.path.join(save_dir, f'{dir_name}-c-{feature_level[:3]}') # features output by context network
    if not os.path.exists(out_dir_c):
        os.makedirs(out_dir_c)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # iterate audios
    for idx, wav_file in enumerate(audio_files, 1):
        file_name = os.path.basename(wav_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')
        # load audio
        # audio, sampling_rate = torchaudio.load(wav_file) # audio: (1, T), float type
        audio, sampling_rate = sf.read(wav_file)
        audio = audio.astype('float32')[np.newaxis, :]
        audio = torch.from_numpy(audio)
        audio = audio.to(device)
        assert sampling_rate == 16000, f'Error: sampling rate ({sampling_rate}) != 16k!'
        t1 = time.time()
        with torch.no_grad():
            z = model.feature_extractor(audio) # (1, C, T), stride: 10ms (100Hz), receptive field: 30ms
            c = model.feature_aggregator(z) # (1, C, T), stride: 10ms (100Hz), receptive field: 801ms (for large version)
        t2 = time.time()
        print(f'Time used for inference: {t2-t1:.1f}s.')

        # save
        z_feature = z.detach().squeeze().t().cpu().numpy()
        c_feature = c.detach().squeeze().t().cpu().numpy()
        # write_feature_to_csv(z_feature, out_dir_z, vid, label_interval, hop_len=10) # hop_len: refer to the paper
        # write_feature_to_csv(c_feature, out_dir_c, vid, label_interval, hop_len=10) # hop_len: refer to the paper
        write_feature_to_npy(z_feature, out_dir_z, vid, label_interval, feature_level, hop_len=10) # hop_len: refer to the paper
        write_feature_to_npy(c_feature, out_dir_c, vid, label_interval, feature_level, hop_len=10) # hop_len: refer to the paper

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


def main(audio_dir, save_dir, feature_level, dir_name=None, overwrite=False, gpu=None):
    print(f'==> Extracting "wav2vec"...')
    
    # infer label interval for specified tasks from audio_dir
    assert feature_level in ['FRAME', 'UTTERANCE']
    if feature_level == 'FRAME': label_interval = 50 # to follow with the facial features
    if feature_level == 'UTTERANCE': label_interval = 500 # to follow with the facial features 
    print(f'==> Note: for "{audio_dir}", the label interval is {label_interval}ms.')

    # in: get audios (assert file extension is '.wav')
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # load model
    device = torch.device(f'cuda:{gpu}')
    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'wav2vec/wav2vec_large.pt')
    cp = torch.load(model_file)
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    model.to(device)
    model.eval()

    # extract features
    extract(audio_files, feature_level=feature_level, model=model, save_dir=save_dir, label_interval=label_interval, dir_name=dir_name, overwrite=overwrite, gpu=gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=7, help='index of gpu')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    args = parser.parse_args()

    feature_level = args.feature_level

    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]

    main(audio_dir, save_dir, feature_level, overwrite=args.overwrite, gpu=args.gpu)