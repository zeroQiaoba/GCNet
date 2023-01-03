# *_*coding:utf-8 *_*
"""
VGGish: https://arxiv.org/abs/1609.09430
official github repo: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
"""
import numpy as np
import six
import tensorflow.compat.v1 as tf # version: 1.15.0 (gpu)
tf.disable_v2_behavior()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import glob
import time
from vggish import vggish_input
from vggish import vggish_params
from vggish import vggish_slim
from util import write_feature_to_csv, write_feature_to_npy
import argparse

# import config
import sys
sys.path.append('../../')
import config

def extract(audio_files, save_dir, label_interval, feature_level, batch_size=2048, gpu=None):
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'vggish/vggish_model.ckpt')
        vggish_slim.load_vggish_slim_checkpoint(sess, model_file)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        for i, audio_file in enumerate(audio_files, 1):
            print(f'Processing "{os.path.basename(audio_file)}" ({i}/{len(audio_files)})...')
            vid = os.path.basename(audio_file)[:-4]
            samples = vggish_input.wavfile_to_examples(audio_file, label_interval / 1000.0) # (sample_size, height(96), width(64))
            sample_size = samples.shape[0]
            # print(f'Sample size: {sample_size}')

            # model inference (max sample size: 6653, will cause OOM. Need to chunk samples.)
            embeddings = []
            num_batches =  int(np.ceil(sample_size / batch_size))
            for i in range(num_batches):
                examples_batch = samples[i*batch_size:min((i+1)*batch_size, sample_size)]
                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})
                embeddings.append(embedding_batch) # [176, 128]
            embeddings = np.row_stack(embeddings)

            # save feature
            # write_feature_to_csv(embeddings, save_dir, vid, label_interval, hop_len=label_interval, win_len=vggish_params.EXAMPLE_WINDOW_SECONDS*1000)
            write_feature_to_npy(embeddings, save_dir, vid, label_interval, feature_level, hop_len=label_interval, win_len=vggish_params.EXAMPLE_WINDOW_SECONDS*1000)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')



def main(audio_dir, save_dir, feature_level, dir_name=None, overwrite=False, gpu=None):
    print(f'==> Extracting "VGGish"...')

    # infer label interval for specified tasks from audio_dir
    assert feature_level in ['FRAME', 'UTTERANCE']
    if feature_level == 'FRAME': label_interval = 50 # to follow with the facial features
    if feature_level == 'UTTERANCE': label_interval = 500 # to follow with the facial features 
    print(f'==> Note: for "{audio_dir}", the label interval is {label_interval}ms.')

    # in: get audios
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # out: check dir
    if dir_name is None:
        dir_name = f'vggish_{feature_level[:3]}'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif overwrite:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # extract features
    extract(audio_files, save_dir, label_interval, feature_level, gpu=gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=7, help='index of gpu')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    args = parser.parse_args()

    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]

    main(audio_dir, save_dir, args.feature_level, gpu=args.gpu, overwrite=args.overwrite)