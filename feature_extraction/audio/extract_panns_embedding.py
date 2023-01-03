"""
PANNs: https://arxiv.org/abs/1912.10211
official github repo: https://github.com/qiuqiangkong/audioset_tagging_cnn
"""
import os
import numpy as np
import argparse
import librosa
import torch
import glob
import time
import math

from panns.models import *
from panns.pytorch_utils import move_data_to_device
from util import write_feature_to_csv, write_feature_to_npy
import argparse

# import config
import sys
sys.path.append('../../')
import config

# supported models
CNN14 = 'Cnn14_mAP=0.431.pth'
CNN14_16K = 'Cnn14_16k_mAP=0.438.pth' # sampling rate: 16000
CNN14_EMB128 = 'Cnn14_emb128_mAP=0.412.pth'
CNN14_EMB512 = 'Cnn14_emb512_mAP=0.420.pth'
ResNet38 = 'ResNet38_mAP=0.434.pth'
WAVEGRAM_LOGMEL_CNN14 = 'Wavegram_Logmel_Cnn14_mAP=0.439.pth'


def segment_waveform(waveform, num_win, num_hop):
    """
    :param waveform: (T,)
    :param num_win: unit: #samples
    :param num_hop: unit: #samples
    :return: numpy array, (batch_size, win_len)
    """
    wav_len = len(waveform)
    seg_num = 1 + int(np.floor((wav_len - num_win) / num_hop))
    waveform_segments = []
    for i in range(seg_num):
        b_idx = i * num_hop
        e_idx = b_idx + num_win
        waveform_segment = waveform[b_idx:e_idx]
        waveform_segments.append(waveform_segment)
    return np.row_stack(waveform_segments)


def extract(model_name, audio_files, save_dir, label_interval, feature_level, win_len=1000,
            batch_size=2048, gpu=None):
    start_time = time.time()
    # setting parameters
    if model_name == CNN14_16K:
        sample_rate = 16000
        window_size = 512
        hop_size = 160
        mel_bins = 64
        fmin = 50
        fmax = 8000
    else:
        sample_rate = 32000
        window_size = 1024
        hop_size = 320
        mel_bins = 64
        fmin = 50
        fmax = 14000
    classes_num = 527 # Note: supervised pre-training on AudioSet (527 classes)

    # construct & load model
    model_class_name = model_name[:model_name.find('_mAP')] # defined in panns/models.py
    Model = eval(model_class_name)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                  classes_num=classes_num)
    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'panns/{model_name}')
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # parallel setting
    if gpu is not None:
        device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
        print('Using CPU.')

    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')
        # load audio
        (waveform, _) = librosa.core.load(audio_file, sr=sample_rate, mono=True) # waveform: [1D]
        ### process for waveform < 1000ms, pad to longer than 1000ms
        if len(waveform) < sample_rate:
            waveform = waveform.tolist()
            waveform = waveform * math.ceil(sample_rate/len(waveform))
            waveform = np.array(waveform)
        num_win, num_hop = int(sample_rate * win_len / 1000.0), int(sample_rate * label_interval / 1000.0)
        waveform_segments = segment_waveform(waveform, num_win, num_hop)
        total_batches = len(waveform_segments)
        num_batches = int(np.ceil(total_batches / batch_size))

        with torch.no_grad():
            model.eval()
            features = []
            # model inference
            for i in range(num_batches):
                batch_waveform_segments = waveform_segments[i*batch_size: min((i+1)*batch_size, total_batches),:] # [175, 16000]
                batch_waveform_segments = move_data_to_device(batch_waveform_segments, device) # 
                batch_output_dict = model(batch_waveform_segments, None)

                assert 'embedding' in batch_output_dict.keys(), 'Error: model has no embedding output!'
                feature = batch_output_dict['embedding'].data.cpu().numpy() # [175, 2048]
                features.append(feature)
            features = np.concatenate(features, axis=0)
            # save feature
            # write_feature_to_csv(features, save_dir, vid, label_interval, hop_len=label_interval, win_len=win_len)
            write_feature_to_npy(features, save_dir, vid, label_interval, feature_level, hop_len=label_interval, win_len=win_len)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


def main(model_name, audio_dir, save_dir, feature_level, dir_name=None, overwrite=False, gpu=None):
    print(f'==> Extracting "PANNs"...')
    # out: save dir
    if dir_name is None:  # use model_name for naming if dir_name is None
        dir_name = model_name[:model_name.find('_mAP')] + f'_{feature_level[:3]}'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # infer label interval for specified tasks from audio_dir
    assert feature_level in ['FRAME', 'UTTERANCE']
    if feature_level == 'FRAME': label_interval = 50 # to follow with the facial features
    if feature_level == 'UTTERANCE': label_interval = 500 # to follow with the facial features 
    print(f'==> Note: for "{audio_dir}", the label interval is {label_interval}ms.')

    # in: get audios
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # extract features
    extract(model_name, audio_files, save_dir, label_interval, feature_level, gpu=gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=7, help='index of gpu')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    args = parser.parse_args()

    model_name = CNN14_16K
    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]

    main(model_name, audio_dir, save_dir, args.feature_level, gpu=args.gpu, overwrite=args.overwrite)