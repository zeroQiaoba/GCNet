import scipy.io as sio
import numpy as np
import math
import os
from numpy.random import shuffle
import tensorflow as tf

class DataSet(object):

    def __init__(self, data, view_number, labels, names):
        """
        Construct a DataSet.
        data: [2, samplenum, dim]
        labels: [samplenum, 1]
        differnt view: self.data[0]; self.data[1];...
        """
        self.data = dict()
        self._num_examples = data[0].shape[0] # samplenum
        self._labels = labels # [samplenum, 1]
        self._names = names   # [samplenum, 1]
        for v_num in range(view_number):
            self.data[str(v_num)] = data[v_num]

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def names(self):
        return self._names

def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


# # './data/animal.mat', 0.8, 1
# def read_data(str_name, ratio, Normal=1):
#     """read data and spilt it train set and test set evenly
#     :param str_name:path and dataname
#     :param ratio:training set ratio
#     :param Normal:do you want normalize
#     :return:dataset and view number
#     """

#     ## X: (viewnum, 1, 1)
#     data = sio.loadmat(str_name)
#     view_number = data['X'].shape[1]
#     X = np.split(data['X'], view_number, axis=1)

#     ## labels: shape: (samplenum, 1), values: [1, 2, ..., classnum]
#     if min(data['gt']) == 0: # 'gt' start from 1 [gt is sorted in decent order]
#         labels = data['gt'] + 1
#     else:
#         labels = data['gt']
#     classes = max(labels)[0]

#     ## gain X_train~labels_test
#     X_train = []
#     X_test = []
#     labels_train = []
#     labels_test = []
#     all_length = 0
#     for c_num in range(1, classes + 1):
#         c_length = np.sum(labels == c_num) # sample number
#         indexes = np.arange(c_length)
#         shuffle(indexes) # gain index list
#         labels_train.extend(labels[all_length + indexes][0:math.floor(c_length * ratio)])
#         labels_test.extend(labels[all_length + indexes][math.floor(c_length * ratio):])

#         # X: (viewnum, 1, 1)
#         X_train_temp = []
#         X_test_temp = []
#         for v_num in range(view_number):
#             X_train_temp.append(X[v_num][0][0].transpose()[all_length + indexes][0:math.floor(c_length * ratio)]) # [2, samplenum, dim]
#             X_test_temp.append(X[v_num][0][0].transpose()[all_length + indexes][math.floor(c_length * ratio):])   # [2, samplenum, dim]
#         if c_num == 1: 
#             X_train = X_train_temp
#             X_test = X_test_temp
#         else: ## append behind
#             for v_num in range(view_number):
#                 X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
#                 X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
#         all_length = all_length + c_length

#     ## normalize each view
#     if (Normal == 1):
#         for v_num in range(view_number):
#             X_train[v_num] = Normalize(X_train[v_num]) # [2, samplenum, dim] 
#             X_test[v_num] = Normalize(X_test[v_num])   # [2, samplenum, dim]

#     # X_train: [2, 8103, 4096]
#     # labels_train: [8103, 1]
#     traindata = DataSet(X_train, view_number, np.array(labels_train))
#     testdata = DataSet(X_test, view_number, np.array(labels_test))
#     return traindata, testdata, view_number





def read_cmumosi_data(data_root, normalize=True):
    """read data and spilt it train set and test set evenly
    :param data_root: data root
    :return:dataset and view number
    """
    view_number = 3
    trn_path = os.path.join(data_root, 'trn.npz')
    val_path = os.path.join(data_root, 'val.npz')
    tst_path = os.path.join(data_root, 'tst.npz')
    
    ### read train data
    X_train = []
    labels_train = []
    names_train = []
    name = np.load(trn_path)['name']
    label = np.load(trn_path)['label']
    audio = np.load(trn_path)['audio']
    video = np.load(trn_path)['video']
    text = np.load(trn_path)['text']

    if min(label) == 0:
        label = label + 1
    classes = max(label)
    # assert classes == 2, f'label needs modified'

    labels_train = label
    names_train = name
    X_train.append(audio)
    X_train.append(video)
    X_train.append(text)

    ### read test data
    X_test = []
    labels_test = []
    names_test = []
    name = np.load(tst_path)['name']
    label = np.load(tst_path)['label']
    audio = np.load(tst_path)['audio']
    video = np.load(tst_path)['video']
    text = np.load(tst_path)['text']

    if min(label) == 0:
        label = label + 1
    classes = max(label)
    # assert classes == 2, f'label needs modified'

    labels_test= label
    names_test = name
    X_test.append(audio)
    X_test.append(video)
    X_test.append(text)

    ## normalize each view
    if normalize:
        for v_num in range(view_number):
            X_train[v_num] = Normalize(X_train[v_num]) # [2, samplenum, dim] 
            X_test[v_num] = Normalize(X_test[v_num])   # [2, samplenum, dim]

    # X_train: [2, 8103, 4096]
    # labels_train: [8103, 1]
    traindata = DataSet(X_train, view_number, np.array(labels_train), names_train)
    testdata = DataSet(X_test, view_number, np.array(labels_test), names_test)
    return traindata, testdata, view_number





def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
