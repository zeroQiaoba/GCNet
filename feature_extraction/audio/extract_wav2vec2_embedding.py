# *_*coding:utf-8 *_*
"""
wav2vec 2.0: https://arxiv.org/abs/2006.11477
official github repo: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
"""
import time
import os
import glob
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from util import frame_audio, write_feature_to_csv, write_feature_to_npy
import argparse
import soundfile as sf

# import config
import sys
sys.path.append('../../')
import config

# supported models
WAV2VEC2_BASE = 'wav2vec2-base'
WAV2VEC2_LARGE = 'wav2vec2-large'
WAV2VEC2_BASE_960H = 'wav2vec2-base-960h' # fine-tuned for ASR on LibriSpeech-960h
WAV2VEC2_LARGE_960H = 'wav2vec2-large-960h' # fine-tuned for ASR on LibriSpeech-960h


def extract(model_name, audio_files, save_dir, label_interval, feature_level, win_len=1000, pool='avg', layer_ids=None,
            batch_size=512, gpu=None):
    start_time = time.time()

    # load model
    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    processor = Wav2Vec2Processor.from_pretrained(model_file)
    model = Wav2Vec2Model.from_pretrained(model_file)
    device = torch.device(f'cuda:{gpu}')
    model.to(device)
    model.eval()

    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        #audio_file = '/data5/lianzheng/deception-detection/dataset/BoxOfLies/subaudio/4.BoL.An.Tr_00_130438_143387.wav'
        file_name = os.path.basename(audio_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')

        ## process for too short ones
        samples, sr = sf.read(audio_file)
        if len(samples) < sr * (win_len / 1000) * 1.2:
            if model_name in [WAV2VEC2_BASE, WAV2VEC2_BASE_960H]:
                feature = np.zeros((1, 768))
            elif model_name == WAV2VEC2_LARGE_960H:
                feature = np.zeros((1, 1024))
            else:
                print ('model_name is not defined!!')
            write_feature_to_npy(feature, save_dir, vid, label_interval, feature_level, hop_len=label_interval, win_len=win_len)
        else:
            # read audio
            audio_inputs = frame_audio(audio_file, win_len, hop_len=label_interval)
            total_batches = len(audio_inputs) # how much 1000ms sample
            num_batches = int(np.ceil(total_batches / batch_size)) # change to batch format
            with torch.no_grad():
                features = []
                for i in range(num_batches):
                    batch_audio_inputs = audio_inputs[i*batch_size:min((i+1)*batch_size, total_batches)]
                    # pre-process
                    input_values = processor(batch_audio_inputs, padding=True, sampling_rate=16000,
                                             return_tensors="pt").input_values # padding for the last frame in audio_inputs (typically its length != win_len)
                    input_values = input_values.to(device) # (B, 16000)
                    # model inference
                    hidden_states = model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
                    feature = torch.stack(hidden_states)[layer_ids].sum(dim=0)  # sum, (B, T, D)
                    if pool == 'avg': # take average
                        feature = torch.mean(feature, dim=1) # operate on dim T, return (B, D) (Now, B is viewed as the temporal dimension)
                    elif pool == 'max':
                        feature = torch.max(feature, dim=1) # operate on dim T, return (B, D) (Now, B is viewed as the temporal dimension)
                    feature = feature.detach().squeeze().cpu().numpy() # (T, D)
                    features.append(feature)
                # concat
                feature = np.concatenate(features, axis=0)
                # save
                # write_feature_to_csv(feature, save_dir, vid, label_interval, hop_len=label_interval, win_len=win_len)
                write_feature_to_npy(feature, save_dir, vid, label_interval, feature_level, hop_len=label_interval, win_len=win_len)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')



def main(model_name, audio_dir, save_dir, feature_level, layer_ids=None, dir_name=None, overwrite=False, gpu=None):
    print(f'==> Extracting "{model_name}"...')

    # layer ids
    if layer_ids is None:
        layer_ids = [-4, -3, -2, -1]
    else:
        assert isinstance(layer_ids, list)

    # out: save dir
    if dir_name is None: # use model_name for naming if dir_name is None
        dir_name = model_name if len(layer_ids) == 1 else f'{model_name}-{len(layer_ids)}'
        dir_name = f'{dir_name}-{feature_level[:3]}'
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
    extract(model_name, audio_files, save_dir, label_interval, feature_level, layer_ids=layer_ids, gpu=gpu)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=7, help='index of gpu')
    parser.add_argument('--model_name', type=str, default='opensmile', help='name of feature extractor')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    args = parser.parse_args()

    # model_name = WAV2VEC2_BASE # OK
    # model_name =  WAV2VEC2_LARGE # OSError: file /data5/emotion-data/Cached/transformers/wav2vec2-large/preprocessor_config.json not found
    # model_name = WAV2VEC2_LARGE_960H # OK
    # model_name =  WAV2VEC2_BASE_960H # OK
    model_name = args.model_name
    feature_level = args.feature_level

    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]
    layer_ids = [-1]

    main(model_name, audio_dir, save_dir, feature_level, layer_ids=layer_ids, overwrite=args.overwrite, gpu=args.gpu)