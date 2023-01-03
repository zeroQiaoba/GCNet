# *_*coding:utf-8 *_*
import os
import re
import numpy as np
import pandas as pd
import soundfile as sf

# 1-D audio to frames, each frame has duration of win_len, the interval between two adjacent frames is hop_len
def frame_audio(audio_file, win_len, hop_len):
    """
    :param audio_file:
    :param win_len: unit: ms
    :param hop_len: unit: ms
    :return: [(num_win,), (num_win,), ...]
    """
    _, sr = sf.read(audio_file)
    assert sr == 16000, f'Error: audio sampling rate ({sr}) != 16k!'
    num_win, num_hop = int(win_len / 1000.0 * sr), int(hop_len / 1000.0 * sr) # unit: number of samples
    blocks = sf.blocks(audio_file, blocksize=num_win, overlap=(num_win-num_hop)) # blocks: generator, Note: overlap is not num_hop!
    frames = []
    for block in blocks: # block: 1-D numpy array, i.e., (win_len,)
        frames.append(block)
    return frames # [(num_win,), (num_win,), ...]


# align extracted feature to corresponding label file and save it
def write_feature_to_csv(feature, save_dir, vid, label_interval, hop_len=10, win_len=None):
    """
    :param feature: a numpy array with dim (T, C)
    :param save_dir: the directory that features will be saved in
    :param vid: the video id associated with extracted feature
    :param label_interval: 250 (unit: ms) for task c1_muse_wilder and c2_muse_sent, 500 (unit: ms) for task c3 and c4
    :param hop_len: the hop length of extracted feature (i.e., temporal interval between two adjacent frames), default is 10 (unit: ms)
    :return:
    """
    try:
        csv_file = os.path.join(save_dir, f'{vid}.csv')
        n_frames, feature_dim = feature.shape  # (T, C)

        # case1 (hop_len == label_interval): directly align timestamps
        if hop_len == label_interval:
            assert win_len != None, 'Error: when label_interval == hop_len (e.g., VGGish or wav2vec2), win_len must be specified!'
            win_len = round(win_len / hop_len) * hop_len
            assert (win_len / 2) % hop_len == 0, 'Error: invalid value for win_len!'
            timestamps = np.arange(n_frames) * hop_len + win_len / 2  # for example, VGGish: 500 ~= win_len / 2 (i.e. 0.975*1000/2)
        else: # case2 (hop_len << label_interval): averaging frame features around each label timestamp
            assert label_interval % hop_len == 0, f'Error: hop_len ({hop_len}) can be divided by label_interval ({label_interval})!'
            n_frames_per_interval = label_interval // hop_len
            head_n_pad_frames = (n_frames_per_interval + 1) // 2  #
            tail_n_pad_frames = n_frames_per_interval - head_n_pad_frames - 1  #
            last_timestamp = (n_frames - 1) * hop_len #
            last_available_label_timestamp = (last_timestamp // label_interval) * label_interval  #
            tail_n_rest_frames = (last_timestamp - last_available_label_timestamp) // hop_len  # avaiable frames after the last available label timestamp
            if tail_n_rest_frames > tail_n_pad_frames:  # no need to pad after tail, only pad zero frames before head
                feature = feature[:-(tail_n_rest_frames - tail_n_pad_frames)]  # truncate
                feature = np.pad(feature, ((head_n_pad_frames, 0), (0, 0)), 'edge')
            else:  # pad zero frames after tail
                tail_n_pad_frames = tail_n_pad_frames - tail_n_rest_frames  # the real number of frames need to pad after tail
                feature = np.pad(feature, ((head_n_pad_frames, tail_n_pad_frames), (0, 0)), 'edge')
            assert len(feature) % n_frames_per_interval == 0, print(f'{len(feature)} <--> {n_frames_per_interval}')
            feature = np.reshape(feature, (-1, n_frames_per_interval, feature_dim))  # (T/n, n, C)
            feature = feature.mean(axis=1)
            timestamps = np.arange(len(feature)) * label_interval

        # construct DataFrame for unaligned features
        meta_columns = ['timestamp', 'segment_id']
        timestamp_column = meta_columns[0]
        columns = [timestamp_column] + [str(i) for i in range(feature_dim)]
        # timestamps = np.arange(len(feature)) * label_interval
        first_timestamp, last_timestamp = timestamps[0], timestamps[-1]
        data = np.column_stack([timestamps, feature])
        df = pd.DataFrame(data=data, columns=columns)
        df[timestamp_column] = df[timestamp_column].astype(np.int64)

        # get corresponding labeled/feature file to align
        task_id = int(re.search('c(\d)_muse_', save_dir).group(1)) # infer the task id from save_dir (naive/unelegant approach)
        if task_id == 2: # for task "c2"
            rel_path = '../au'  # use csv file in "au" feature as reference beacause of there is no timestamp in the label file
        elif task_id == 4: # for task "c4"
            rel_path = '../../label_segments/anno12_EDA' # no arousal label for this task
        else:
            rel_path = '../../label_segments/arousal'
        label_dir = os.path.abspath(os.path.join(save_dir, rel_path))
        assert os.path.exists(label_dir), f'Error:  label dir "{label_dir}" does not exist!'
        label_file = os.path.join(label_dir, f'{vid}.csv')
        df_label = pd.read_csv(label_file)
        metas = df_label[meta_columns].values
        label_first_timestamp, label_last_timestamp = metas[0, 0], metas[-1, 0]

        # pad before head according to the first label timestamp if necessary
        if first_timestamp > label_first_timestamp:
            n_pad_frames = int((first_timestamp - label_first_timestamp) / label_interval)
            pad_timestamps = np.arange(label_first_timestamp, first_timestamp, label_interval)
            print(f'Note: label first timestamp ({label_first_timestamp}) < feature first timestamp ({first_timestamp}). '
                  f'Pad first frame (<--) of feature for timestamps: {pad_timestamps.tolist()}.')
            pad_features = np.tile(df.iloc[0].values[1:], (n_pad_frames, 1))
            pad_data = np.column_stack([pad_timestamps, pad_features])
            data = np.row_stack([pad_data, data])

        # pad after tail according to the last label timestamp if necessary
        if last_timestamp < label_last_timestamp:
            n_pad_frames = int((label_last_timestamp - last_timestamp) / label_interval)
            pad_timestamps = np.arange(last_timestamp, label_last_timestamp, label_interval) + label_interval
            print(f'Note: feature last timestamp ({last_timestamp}) < label last timestamp ({label_last_timestamp}). '
                  f'Pad last frame (-->) of feature for timestamps: {pad_timestamps.tolist()}.')
            pad_features = np.tile(df.iloc[-1].values[1:], (n_pad_frames, 1))
            pad_data = np.column_stack([pad_timestamps, pad_features])
            data = np.row_stack([data, pad_data])

        df = pd.DataFrame(data, columns=columns)
        df[timestamp_column] = df[timestamp_column].astype(np.int64)
        first_timestamp, last_timestamp = df.iloc[0, 0], df.iloc[-1, 0]
        assert first_timestamp <= label_first_timestamp and last_timestamp >= label_last_timestamp, 'Error!'
        # align features to label timestamps
        label_aligned_features = df[df[timestamp_column].isin(df_label[timestamp_column])].values[:, 1:]
        columns = meta_columns + [str(i) for i in range(feature_dim)]
        data = np.column_stack([metas, label_aligned_features])
        df = pd.DataFrame(data, columns=columns)
        df[meta_columns] = df[meta_columns].astype(np.int64)
        df.to_csv(csv_file, index=False)
    except Exception as e:
        print(f'When processing "{vid}.wav" exception "{e}" occurred!')




# align extracted feature to corresponding label file and save it
def write_feature_to_npy(feature, save_dir, vid, label_interval, feature_level, hop_len=10, win_len=None):
    """
    :param feature: a numpy array with dim (T, C)
    :param save_dir: the directory that features will be saved in
    :param vid: the video id associated with extracted feature
    :param label_interval: 250 (unit: ms) for task c1_muse_wilder and c2_muse_sent, 500 (unit: ms) for task c3 and c4
    :param hop_len: the hop length of extracted feature (i.e., temporal interval between two adjacent frames), default is 10 (unit: ms)
    :return:
    """
    try:
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        
        if feature_level == 'UTTERANCE':
            feature = feature.squeeze() # [C,]
            if len(feature.shape) != 1: # change [T, C] => [C,]
                feature = np.mean(feature, axis=0)
            np.save(csv_file, feature)
            return

        ## When feature_level is 'FRAME'
        n_frames, feature_dim = feature.shape  # (T, C)
        # case1 (hop_len == label_interval): directly align timestamps
        if hop_len == label_interval:
            assert win_len != None, 'Error: when label_interval == hop_len (e.g., VGGish or wav2vec2), win_len must be specified!'
            # win_len = round(win_len / hop_len) * hop_len
            # assert (win_len / 2) % hop_len == 0, 'Error: invalid value for win_len!'
            # timestamps = np.arange(n_frames) * hop_len + win_len / 2  # for example, VGGish: 500 ~= win_len / 2 (i.e. 0.975*1000/2)
            np.save(csv_file, feature)
        else: # case2 (hop_len << label_interval): averaging frame features around each label timestamp
            assert label_interval % hop_len == 0, f'Error: hop_len ({hop_len}) can be divided by label_interval ({label_interval})!'
            n_frames_per_interval = label_interval // hop_len
            head_n_pad_frames = (n_frames_per_interval + 1) // 2  #
            tail_n_pad_frames = n_frames_per_interval - head_n_pad_frames - 1  #
            last_timestamp = (n_frames - 1) * hop_len #
            last_available_label_timestamp = (last_timestamp // label_interval) * label_interval  #
            tail_n_rest_frames = (last_timestamp - last_available_label_timestamp) // hop_len  # avaiable frames after the last available label timestamp
            if tail_n_rest_frames > tail_n_pad_frames:  # no need to pad after tail, only pad zero frames before head
                feature = feature[:-(tail_n_rest_frames - tail_n_pad_frames)]  # truncate
                feature = np.pad(feature, ((head_n_pad_frames, 0), (0, 0)), 'edge')
            else:  # pad zero frames after tail
                tail_n_pad_frames = tail_n_pad_frames - tail_n_rest_frames  # the real number of frames need to pad after tail
                feature = np.pad(feature, ((head_n_pad_frames, tail_n_pad_frames), (0, 0)), 'edge')
            assert len(feature) % n_frames_per_interval == 0, print(f'{len(feature)} <--> {n_frames_per_interval}')
            feature = np.reshape(feature, (-1, n_frames_per_interval, feature_dim))  # (T/n, n, C)
            feature = feature.mean(axis=1)
            timestamps = np.arange(len(feature)) * label_interval
            np.save(csv_file, feature)

    except Exception as e:
        print(f'When processing "{vid}.wav" exception "{e}" occurred!')