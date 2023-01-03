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
    for ii, name in tqdm.tqdm(enumerate(names)):
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
def change_feat_format_cmumosei():

    label_pkl = '../dataset/CMUMOSEI/CMUMOSEI_features_raw_2way.pkl'
    feat_root = '../dataset/CMUMOSEI/features'
    save_root = './CMUMOSEI_features_2021'
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
        label = []
        int2name = []
        for vid in tqdm.tqdm(item1):
            int2name.extend(videoIDs[vid])
            label.extend(videoLabels[vid])
            for ii in range(len(videoIDs[vid])):
                name = videoIDs[vid][ii]
                featA = name2featA[name]
                featV = name2featV[name]
                featL = name2featL[name]
                all_A.append(featA)
                all_V.append(featV)
                all_L.append(featL)
        all_A = np.array(all_A)
        all_V = np.array(all_V)
        all_L = np.array(all_L)

        save_path = f"{save_root}/A/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_A)

        save_path = f"{save_root}/V/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_V)

        save_path = f"{save_root}/L/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_L)

        save_path = f"{save_root}/target/1/{item2}_label.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, label)

        save_path = f"{save_root}/target/1/{item2}_int2name.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, int2name)



#########################################################
## Process for iemocapfour
#########################################################
def change_feat_format_iemocapfour():
    label_pkl = '../dataset/IEMOCAP/IEMOCAP_features_raw_4way.pkl'
    feat_root = '../dataset/IEMOCAP/features'
    save_root = './IEMOCAPFOUR_features_2021'
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
            label = []
            int2name = []
            for vid in tqdm.tqdm(item1):
                int2name.extend(videoIDs[vid])
                label.extend(videoLabels[vid])
                for jj in range(len(videoIDs[vid])):
                    name = videoIDs[vid][jj]
                    featA = name2featA[name]
                    featV = name2featV[name]
                    featL = name2featL[name]
                    all_A.append(featA)
                    all_V.append(featV)
                    all_L.append(featL)
            all_A = np.array(all_A)
            all_V = np.array(all_V)
            all_L = np.array(all_L)

            save_path = f"{save_root}/A/{ii+1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_A)

            save_path = f"{save_root}/V/{ii+1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_V)

            save_path = f"{save_root}/L/{ii+1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_L)

            save_path = f"{save_root}/target/{ii+1}/{item2}_label.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, label)

            save_path = f"{save_root}/target/{ii+1}/{item2}_int2name.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, int2name)



#########################################################
## Process for iemocapfour
#########################################################
def change_feat_format_iemocapsix():
    label_pkl = '../dataset/IEMOCAP/IEMOCAP_features_raw_6way.pkl'
    feat_root = '../dataset/IEMOCAP/features'
    save_root = './IEMOCAPSIX_features_2021'
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
            label = []
            int2name = []
            for vid in tqdm.tqdm(item1):
                int2name.extend(videoIDs[vid])
                label.extend(videoLabels[vid])
                for jj in range(len(videoIDs[vid])):
                    name = videoIDs[vid][jj]
                    featA = name2featA[name]
                    featV = name2featV[name]
                    featL = name2featL[name]
                    all_A.append(featA)
                    all_V.append(featV)
                    all_L.append(featL)
            all_A = np.array(all_A)
            all_V = np.array(all_V)
            all_L = np.array(all_L)

            save_path = f"{save_root}/A/{ii+1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_A)

            save_path = f"{save_root}/V/{ii+1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_V)

            save_path = f"{save_root}/L/{ii+1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_L)

            save_path = f"{save_root}/target/{ii+1}/{item2}_label.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, label)

            save_path = f"{save_root}/target/{ii+1}/{item2}_int2name.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, int2name)



if __name__ == '__main__':
    import fire
    fire.Fire()


