import scipy.io as sio
import numpy as np
import math
import os
from numpy.random import shuffle
import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder

def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def read_cmumosi_data(data_root, normalize=True):
    """read data and spilt it train set and test set evenly
    :param data_root: data root
    :return:dataset and view number
    """

    trn_path = os.path.join(data_root, 'trn.npz')
    val_path = os.path.join(data_root, 'val.npz')
    tst_path = os.path.join(data_root, 'tst.npz')
    
    ### read train data
    X_train = []
    names_train = np.load(trn_path)['name']
    labels_train = np.load(trn_path)['label']
    audio = np.load(trn_path)['audio']
    video = np.load(trn_path)['video']
    text = np.load(trn_path)['text']
    X_train.append(audio)
    X_train.append(video)
    X_train.append(text)


    ### read train data
    X_val = []
    names_val = np.load(val_path)['name']
    labels_val = np.load(val_path)['label']
    audio = np.load(val_path)['audio']
    video = np.load(val_path)['video']
    text = np.load(val_path)['text']
    X_val.append(audio)
    X_val.append(video)
    X_val.append(text)


    ### read test data
    X_test = []
    names_test = np.load(tst_path)['name']
    labels_test = np.load(tst_path)['label']
    audio = np.load(tst_path)['audio']
    video = np.load(tst_path)['video']
    text = np.load(tst_path)['text']
    X_test.append(audio)
    X_test.append(video)
    X_test.append(text)


    ## normalize each view
    if normalize:
        view_number = 3
        for v_num in range(view_number):
            X_train[v_num] = Normalize(X_train[v_num]) # [2, samplenum, dim]
            X_val[v_num] = Normalize(X_val[v_num]) # [2, samplenum, dim]
            X_test[v_num] = Normalize(X_test[v_num])   # [2, samplenum, dim]

    return X_train, labels_train, X_val, labels_val, X_test, labels_test, names_test


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    one_rate = 1-missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum=2] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num)) # [samplenum, viewnum=2] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        # print (f'first ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        # print (f'second ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        # print (f'third ratio: {ratio}')
        error = abs(one_rate - ratio)
        
    return matrix


def padmeanV1(X_train, Sn_train, X_valid, Sn_valid, X_test, Sn_test):
    
    # gain number
    trainNum = len(X_train)
    validNum = len(X_valid)
    testNum = len(X_test)

    # (X, Sn) -> 计算可见样本的均值
    # X: [samplenum, dim]
    # Sn: [samplenum, ]
    X = np.concatenate([X_train, X_valid, X_test], axis=0)
    Sn = np.concatenate([Sn_train, Sn_valid, Sn_test], axis=0)
    meanfeat = np.mean(X[Sn==1], axis=0)

    # (X, Sn, meanfeat) -> Xnew
    X_new = []
    for ii in range(len(X)):
        if Sn[ii] == 0:
            X_new.append(meanfeat)
        else:
            X_new.append(X[ii])
    X_new = np.array(X_new)

    # split into train, valid, test
    X_trainNew = X_new[:trainNum, :]
    X_validNew = X_new[trainNum:trainNum+validNum, :]
    X_testNew = X_new[trainNum+validNum:, :]
    assert len(X_testNew) == testNum

    return X_trainNew, X_validNew, X_testNew


def padmeanV2(X_train, Sn_train, y_train, X_valid, Sn_valid, y_valid, X_test, Sn_test, y_test):
    
    # gain number
    trainNum = len(X_train)
    validNum = len(X_valid)
    testNum = len(X_test)

    # 可见样本中，每类的均值
    X = np.concatenate([X_train, X_valid, X_test], axis=0)
    Sn = np.concatenate([Sn_train, Sn_valid, Sn_test], axis=0)
    y = np.concatenate([y_train, y_valid, y_test], axis=0)
    n_classes = max(y) - min(y) + 1
    cls2feat = {}
    for ii in range(len(X)):
        y_sample = y[ii]
        Sn_sample = Sn[ii]
        X_sample = X[ii]
        if y_sample not in cls2feat: cls2feat[y_sample] = []
        if Sn_sample == 1:
            cls2feat[y_sample].append(X_sample)

    cls2mean = {}
    for label in cls2feat:
        feats = cls2feat[label]
        meanfeat = np.mean(feats, axis=0)
        cls2mean[label] = meanfeat


    # calculate X_new
    X_new = []
    for ii in range(len(X)):
        y_sample = y[ii]
        Sn_sample = Sn[ii]
        X_sample = X[ii]
        if Sn_sample == 0:
            X_new.append(cls2mean[y_sample])
        else:
            X_new.append(X_sample)
    X_new = np.array(X_new)

    # split into train, valid, test
    X_trainNew = X_new[:trainNum, :]
    X_validNew = X_new[trainNum:trainNum+validNum, :]
    X_testNew = X_new[trainNum+validNum:, :]
    assert len(X_testNew) == testNum

    return X_trainNew, X_validNew, X_testNew


def padmeanV3(X_train, Sn_train, X_valid, Sn_valid, X_test, Sn_test):
    
    # gain number
    trainNum = len(X_train)
    validNum = len(X_valid)
    testNum = len(X_test)

    # (X, Sn) -> 计算可见样本的均值
    # X: [samplenum, dim]
    # Sn: [samplenum, ]
    X = np.concatenate([X_train, X_valid, X_test], axis=0)
    Sn = np.concatenate([Sn_train, Sn_valid, Sn_test], axis=0)
    meanfeat = np.mean(X_train[Sn_train==1], axis=0)

    # (X, Sn, meanfeat) -> Xnew
    X_new = []
    for ii in range(len(X)):
        if Sn[ii] == 0:
            X_new.append(meanfeat)
        else:
            X_new.append(X[ii])
    X_new = np.array(X_new)

    # split into train, valid, test
    X_trainNew = X_new[:trainNum, :]
    X_validNew = X_new[trainNum:trainNum+validNum, :]
    X_testNew = X_new[trainNum+validNum:, :]
    assert len(X_testNew) == testNum

    return X_trainNew, X_validNew, X_testNew


def padmeanV4(X_train, Sn_train, y_train, X_valid, Sn_valid, y_valid, X_test, Sn_test, y_test):
    
    # gain number
    trainNum = len(X_train)
    validNum = len(X_valid)
    testNum = len(X_test)

    # 可见样本中，每类的均值
    X = np.concatenate([X_train, X_valid, X_test], axis=0)
    Sn = np.concatenate([Sn_train, Sn_valid, Sn_test], axis=0)
    y = np.concatenate([y_train, y_valid, y_test], axis=0)
    n_classes = max(y) - min(y) + 1
    cls2feat = {}
    for ii in range(len(X_train)):
        y_sample = y_train[ii]
        Sn_sample = Sn_train[ii]
        X_sample = X_train[ii]
        if y_sample not in cls2feat: cls2feat[y_sample] = []
        if Sn_sample == 1:
            cls2feat[y_sample].append(X_sample)

    cls2mean = {}
    for label in cls2feat:
        feats = cls2feat[label]
        meanfeat = np.mean(feats, axis=0)
        cls2mean[label] = meanfeat


    # calculate X_new
    X_new = []
    for ii in range(len(X)):
        y_sample = y[ii]
        Sn_sample = Sn[ii]
        X_sample = X[ii]
        if Sn_sample == 0:
            X_new.append(cls2mean[y_sample])
        else:
            X_new.append(X_sample)
    X_new = np.array(X_new)

    # split into train, valid, test
    X_trainNew = X_new[:trainNum, :]
    X_validNew = X_new[trainNum:trainNum+validNum, :]
    X_testNew = X_new[trainNum+validNum:, :]
    assert len(X_testNew) == testNum

    return X_trainNew, X_validNew, X_testNew