import numpy as np
from sklearn.preprocessing import OneHotEncoder


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def vote(lsd1, lsd2, label, n=1):
    """Sometimes the prediction accuracy will be higher in this way.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :param n: Similar to K in k-nearest neighbors algorithm
    :return: Predicted label
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    gt_list = []
    label = label.reshape(len(label), 1)
    for num in range(n):
        F_h_h_argmax = np.argmax(F_h_h, axis=1)
        F_h_h_onehot = convert_to_one_hot(F_h_h_argmax, len(label))
        F_h_h = F_h_h - np.multiply(F_h_h, F_h_h_onehot)
        gt_list.append(np.dot(F_h_h_onehot, label))
    gt_ = np.array(gt_list).transpose(2, 1, 0)[0].astype(np.int64)
    count_list = []
    count_list.append([np.argmax(np.bincount(gt_[i])) for i in range(lsd2.shape[0])])
    gt_pre = np.array(count_list)
    return gt_pre.transpose()

def ave(lsd1, lsd2, label):
    """In most cases, this method is used to predict the highest accuracy.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted label
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1)) # 每个测试样本和所有训练样本的距离
    label = label.reshape(len(label), 1) - 1
    enc = OneHotEncoder()
    a = enc.fit_transform(label)
    label_onehot = a.toarray()               # 每个测试样本的onehot标签
    label_num = np.sum(label_onehot, axis=0) # 没有样本数量【训练集】
    F_h_h_sum = np.dot(F_h_h, label_onehot)
    F_h_h_mean = F_h_h_sum / label_num       # 每个类的中性距离
    label_pre = np.argmax(F_h_h_mean, axis=1) + 1
    return label_pre                         # 计算预测标签结果
