import os
import time
import glob
import tqdm
import pickle
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


## gain name2features
def read_data(label_path, feature_root):

    ## gain (names, speakers)
    names = []
    speakers = []
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid = pickle.load(open(label_path, "rb"), encoding='latin1')
    vids = sorted(list(trainVid | testVid))
    for ii, vid in enumerate(vids):
        uids_video = videoIDs[vid]
        spks_video = videoSpeakers[vid]
        names.extend(uids_video)
        speakers.extend(spks_video)

    ## (names, speakers) => features
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        speaker = speakers[ii]
        feature_dir = glob.glob(os.path.join(feature_root, name+'*'))
        assert len(feature_dir) == 1
        feature_path = feature_dir[0]

        feature = {'F':[], 'M':[]}
        if feature_path.endswith('.npy'): # audio/text => belong to speaker
            single_feature = np.load(feature_path)
            single_feature = single_feature.squeeze() # [Dim, ] or [Time, Dim]
            feature[speaker].append(single_feature)
            feature_dim = max(feature_dim, single_feature.shape[-1])
        else: ## exists dir, faces => belong to speaker in 'facename'
            facenames = os.listdir(feature_path)
            for facename in sorted(facenames):
                assert facename.find('F') >= 0 or facename.find('M') >= 0
                facefeat = np.load(os.path.join(feature_path, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                if facename.find('F') >= 0:
                    feature['F'].append(facefeat)
                else:
                    feature['M'].append(facefeat)

        for speaker in feature:
            single_feature = np.array(feature[speaker]).squeeze()
            if len(single_feature) == 0:
                single_feature = np.zeros((feature_dim, ))
            elif len(single_feature.shape) == 2:
                single_feature = np.mean(single_feature, axis=0)
            feature[speaker] = single_feature
        features.append(feature)

    ## save (names, features)
    print (f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats = {}
    for ii in range(len(names)):
        name2feats[names[ii]] = features[ii]

    return name2feats, feature_dim



class IEMOCAPDataset(Dataset):

    def __init__(self, label_path, audio_root, text_root, video_root):

        ## read utterance feats
        name2audio, adim = read_data(label_path, audio_root)
        name2text, tdim = read_data(label_path, text_root)
        name2video, vdim = read_data(label_path, video_root)
        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim

        ## gain video feats
        self.max_len = -1
        self.videoAudioHost = {}
        self.videoTextHost = {}
        self.videoVisualHost = {}
        self.videoAudioGuest = {}
        self.videoTextGuest = {}
        self.videoVisualGuest = {}
        self.videoLabelsNew = {}
        self.videoSpeakersNew = {}
        # labelmap = {'lie': 0, 'truthful':1} => already int
        speakermap = {'F':0, 'M': 1}
        self.videoIDs, self.videoLabels, self.videoSpeakers, self.videoSentences, self.trainVid, self.testVid = pickle.load(open(label_path, "rb"), encoding='latin1')

        self.vids = sorted(list(self.trainVid | self.testVid))
        for ii, vid in enumerate(self.vids):
            uids = self.videoIDs[vid]
            labels = self.videoLabels[vid]
            speakers = self.videoSpeakers[vid]

            self.max_len = max(self.max_len, len(uids))
            self.videoAudioHost[vid] = []
            self.videoTextHost[vid] = []
            self.videoVisualHost[vid] = []
            self.videoAudioGuest[vid] = []
            self.videoTextGuest[vid] = []
            self.videoVisualGuest[vid] = []
            self.videoLabelsNew[vid] = []
            self.videoSpeakersNew[vid] = []
            for ii, uid in enumerate(uids):
                self.videoAudioHost[vid].append(name2audio[uid]['F'])
                self.videoTextHost[vid].append(name2text[uid]['F'])
                self.videoVisualHost[vid].append(name2video[uid]['F'])
                self.videoAudioGuest[vid].append(name2audio[uid]['M'])
                self.videoTextGuest[vid].append(name2text[uid]['M'])
                self.videoVisualGuest[vid].append(name2video[uid]['M'])
                self.videoLabelsNew[vid].append(labels[ii])
                self.videoSpeakersNew[vid].append(speakermap[speakers[ii]])
            self.videoAudioHost[vid] = np.array(self.videoAudioHost[vid])
            self.videoTextHost[vid] = np.array(self.videoTextHost[vid])
            self.videoVisualHost[vid] = np.array(self.videoVisualHost[vid])
            self.videoAudioGuest[vid] = np.array(self.videoAudioGuest[vid])
            self.videoTextGuest[vid] = np.array(self.videoTextGuest[vid])
            self.videoVisualGuest[vid] = np.array(self.videoVisualGuest[vid])
            self.videoLabelsNew[vid] = np.array(self.videoLabelsNew[vid])
            self.videoSpeakersNew[vid] = np.array(self.videoSpeakersNew[vid])


    ## return host(A, T, V) and guest(A, T, V)
    def __getitem__(self, index):
        vid = self.vids[index]
        return torch.FloatTensor(self.videoAudioHost[vid]),\
               torch.FloatTensor(self.videoTextHost[vid]),\
               torch.FloatTensor(self.videoVisualHost[vid]),\
               torch.FloatTensor(self.videoAudioGuest[vid]),\
               torch.FloatTensor(self.videoTextGuest[vid]),\
               torch.FloatTensor(self.videoVisualGuest[vid]),\
               torch.FloatTensor(self.videoSpeakersNew[vid]),\
               torch.FloatTensor([1]*len(self.videoLabelsNew[vid])),\
               torch.LongTensor(self.videoLabelsNew[vid]),\
               vid


    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print (f'max seqlen: {self.max_len}')
        return self.max_len

    def collate_fn(self, data):
        datnew = []
        dat = pd.DataFrame(data)
        for i in dat: # row index
            if i<=5: 
                datnew.append(pad_sequence(dat[i])) # pad
            elif i<=8:
                datnew.append(pad_sequence(dat[i], True)) # reverse
            else:
                datnew.append(dat[i].tolist()) # origin
        return datnew