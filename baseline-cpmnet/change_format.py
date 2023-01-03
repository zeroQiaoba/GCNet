import re
import os
import copy
import tqdm
import glob
import json
import math
import shutil
import random
import pickle
import numpy as np
import soundfile as sf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def name2feat(feature_root):

    ## gain (names)
    names = os.listdir(feature_root)

    ## (names, speakers) => features
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        feature = []
        feature_path = os.path.join(feature_root, name) # folder or npy
        # print (f'process name: {name}  {ii+1}/{len(names)}')
        if os.path.isfile(feature_path): # for .npy
            single_feature = np.load(feature_path)
            single_feature = single_feature.squeeze() # [Dim, ] or [Time, Dim]
            feature.append(single_feature)
            feature_dim = max(feature_dim, single_feature.shape[-1])
        else: ## exists dir, faces
            facenames = os.listdir(feature_path)
            for facename in sorted(facenames):
                facefeat = np.load(os.path.join(feature_path, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                feature.append(facefeat)
        # sequeeze features
        single_feature = np.array(feature).squeeze()
        if len(single_feature) == 0:
            single_feature = np.zeros((feature_dim, ))
        elif len(single_feature.shape) == 2:
            single_feature = np.mean(single_feature, axis=0)
        features.append(single_feature)

    ## save (names, features)
    print (f'Input feature {os.path.basename(feature_root)} ===> dim is {feature_dim}; No. sample is {len(names)}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats = {}
    for ii in range(len(names)):
        name = names[ii]
        if name.endswith('.npy') or name.endswith('.npz'):
            name = name[:-4]
        name2feats[name] = features[ii]

    ## return name2feats
    return name2feats



#########################################################
## Process for cmumosi
#########################################################
def change_feat_format_cmumosi():

    label_pkl = '../dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'
    feat_root = '../dataset/CMUMOSI/features'
    save_root = './data/cmumosi'
    nameA = 'wav2vec-large-c-UTT'
    nameV = 'manet_UTT'
    nameL = 'deberta-large-4'
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, valVid, testVid = pickle.load(open(label_pkl, "rb"), encoding='latin1')
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)
    name2featA = name2feat(featrootA)
    name2featV = name2feat(featrootV)
    name2featL = name2feat(featrootL)


    for (item1, item2) in [(trainVid, 'trn'), (valVid, 'val'), (testVid, 'tst')]:
        all_A = []
        all_V = []
        all_L = []
        all_labels = []
        all_names = []

        ## read all_names and all_labels
        for vid in tqdm.tqdm(item1):
            names = videoIDs[vid]
            labels = videoLabels[vid]

            ## change labels
            for ii, label in enumerate(labels):
                if label == 0:
                    continue
                elif label > 0:
                    all_labels.append(1)
                    all_names.append(names[ii])
                else:
                    all_labels.append(0)
                    all_names.append(names[ii])

        ## all_names -> feat
        for name in all_names:
            featA = name2featA[name]
            featV = name2featV[name]
            featL = name2featL[name]
            all_A.append(featA)
            all_V.append(featV)
            all_L.append(featL)
        all_A = np.array(all_A).astype('float32')
        all_V = np.array(all_V).astype('float32')
        all_L = np.array(all_L).astype('float32')

        ## save path
        save_path = f"{save_root}/1/{item2}.npz"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.savez(save_path, name = all_names, label = all_labels, audio = all_A, video = all_V, text = all_L)




#########################################################
## Process for cmumosi
#########################################################
def change_feat_format_cmumosei():

    label_pkl = '../dataset/CMUMOSEI/CMUMOSEI_features_raw_2way.pkl'
    feat_root = '../dataset/CMUMOSEI/features'
    save_root = './data/cmumosei'
    nameA = 'wav2vec-large-c-UTT'
    nameV = 'manet_UTT'
    nameL = 'deberta-large-4-UTT'
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, valVid, testVid = pickle.load(open(label_pkl, "rb"), encoding='latin1')
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)
    name2featA = name2feat(featrootA)
    name2featV = name2feat(featrootV)
    name2featL = name2feat(featrootL)


    for (item1, item2) in [(trainVid, 'trn'), (valVid, 'val'), (testVid, 'tst')]:
        all_A = []
        all_V = []
        all_L = []
        all_labels = []
        all_names = []

        ## read all_names and all_labels
        for vid in tqdm.tqdm(item1):
            names = videoIDs[vid]
            labels = videoLabels[vid]

            ## change labels
            for ii, label in enumerate(labels):
                if label == 0:
                    continue
                elif label > 0:
                    all_labels.append(1)
                    all_names.append(names[ii])
                else:
                    all_labels.append(0)
                    all_names.append(names[ii])

        ## all_names -> feat
        for name in all_names:
            featA = name2featA[name]
            featV = name2featV[name]
            featL = name2featL[name]
            all_A.append(featA)
            all_V.append(featV)
            all_L.append(featL)
        all_A = np.array(all_A).astype('float32')
        all_V = np.array(all_V).astype('float32')
        all_L = np.array(all_L).astype('float32')

        ## save path
        save_path = f"{save_root}/1/{item2}.npz"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.savez(save_path, name = all_names, label = all_labels, audio = all_A, video = all_V, text = all_L)



#########################################################
## Process for iemocapfour
#########################################################
def change_feat_format_iemocapfour():
    label_pkl = '../dataset/IEMOCAP/IEMOCAP_features_raw_4way.pkl'
    feat_root = '../dataset/IEMOCAP/features'
    save_root = './data/iemocapfour'
    nameA = 'wav2vec-large-c-UTT'
    nameV = 'manet_UTT'
    nameL = 'deberta-large-4-UTT'
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid = pickle.load(open(label_pkl, "rb"), encoding='latin1')
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)
    name2featA = name2feat(featrootA)
    name2featV = name2feat(featrootV)
    name2featL = name2feat(featrootL)

    ## generate five folders
    num_folder = 5
    vids = sorted(list(trainVid | testVid))

    session_to_idx = {}
    for idx, vid in enumerate(vids):
        session = int(vid[4]) - 1
        if session not in session_to_idx: session_to_idx[session] = []
        session_to_idx[session].append(idx)
    assert len(session_to_idx) == num_folder, f'Must split into five folder'

    train_test_idxs = []
    for ii in range(num_folder): # ii in [0, 4]
        test_idxs = session_to_idx[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii: train_idxs.extend(session_to_idx[jj])
        train_test_idxs.append([train_idxs, test_idxs])

    ## for each folder
    for ii in range(len(train_test_idxs)):
        train_idxs = train_test_idxs[ii][0]
        test_idxs = train_test_idxs[ii][1]
        trainVid = np.array(vids)[train_idxs]
        testVid = np.array(vids)[test_idxs]

        for (item1, item2) in [(trainVid, 'trn'), (testVid, 'val'), (testVid, 'tst')]:
            ## change to utterance-level feats
            all_A = []
            all_V = []
            all_L = []
            all_labels = []
            all_names = []
            for vid in tqdm.tqdm(item1):
                all_names.extend(videoIDs[vid])
                all_labels.extend(videoLabels[vid])

            ## all_names -> feat
            for name in all_names:
                featA = name2featA[name]
                featV = name2featV[name]
                featL = name2featL[name]
                all_A.append(featA)
                all_V.append(featV)
                all_L.append(featL)
            all_A = np.array(all_A).astype('float32')
            all_V = np.array(all_V).astype('float32')
            all_L = np.array(all_L).astype('float32')

            save_path = f"{save_root}/{ii+1}/{item2}.npz"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.savez(save_path, name = all_names, label = all_labels, audio = all_A, video = all_V, text = all_L)



#########################################################
## Process for iemocapfour
#########################################################
def change_feat_format_iemocapsix():
    label_pkl = '../dataset/IEMOCAP/IEMOCAP_features_raw_6way.pkl'
    feat_root = '../dataset/IEMOCAP/features'
    save_root = './data/iemocapsix'
    nameA = 'wav2vec-large-c-UTT'
    nameV = 'manet_UTT'
    nameL = 'deberta-large-4-UTT'
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid = pickle.load(open(label_pkl, "rb"), encoding='latin1')
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)
    name2featA = name2feat(featrootA)
    name2featV = name2feat(featrootV)
    name2featL = name2feat(featrootL)

    ## generate five folders
    num_folder = 5
    vids = sorted(list(trainVid | testVid))

    session_to_idx = {}
    for idx, vid in enumerate(vids):
        session = int(vid[4]) - 1
        if session not in session_to_idx: session_to_idx[session] = []
        session_to_idx[session].append(idx)
    assert len(session_to_idx) == num_folder, f'Must split into five folder'

    train_test_idxs = []
    for ii in range(num_folder): # ii in [0, 4]
        test_idxs = session_to_idx[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii: train_idxs.extend(session_to_idx[jj])
        train_test_idxs.append([train_idxs, test_idxs])

    ## for each folder
    for ii in range(len(train_test_idxs)):
        train_idxs = train_test_idxs[ii][0]
        test_idxs = train_test_idxs[ii][1]
        trainVid = np.array(vids)[train_idxs]
        testVid = np.array(vids)[test_idxs]

        for (item1, item2) in [(trainVid, 'trn'), (testVid, 'val'), (testVid, 'tst')]:
            ## change to utterance-level feats
            all_A = []
            all_V = []
            all_L = []
            all_labels = []
            all_names = []
            for vid in tqdm.tqdm(item1):
                all_names.extend(videoIDs[vid])
                all_labels.extend(videoLabels[vid])

            ## all_names -> feat
            for name in all_names:
                featA = name2featA[name]
                featV = name2featV[name]
                featL = name2featL[name]
                all_A.append(featA)
                all_V.append(featV)
                all_L.append(featL)
            all_A = np.array(all_A).astype('float32')
            all_V = np.array(all_V).astype('float32')
            all_L = np.array(all_L).astype('float32')

            save_path = f"{save_root}/{ii+1}/{item2}.npz"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.savez(save_path, name = all_names, label = all_labels, audio = all_A, video = all_V, text = all_L)


if __name__ == '__main__':
    import fire
    fire.Fire()