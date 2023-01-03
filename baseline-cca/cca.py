import os
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import time
import numpy as np
from numpy.random import randint

from util import *

import sys
sys.path.append('../')
import config

class CCA:
    def __init__(self, n_components=1, r1=1e-4, r2=1e-4):
        self.n_components = n_components
        self.r1 = r1
        self.r2 = r2
        self.w = [None, None]
        self.m = [None, None]

    # (50000, 784)
    def fit(self, X1, X2):
        N = X1.shape[0]  # 50000
        f1 = X1.shape[1] # 784
        f2 = X2.shape[1] # 784

        self.m[0] = np.mean(X1, axis=0, keepdims=True) # [1, f1]
        self.m[1] = np.mean(X2, axis=0, keepdims=True)
        H1bar = X1 - self.m[0] # zero mean X1
        H2bar = X2 - self.m[1] # zero mean X2

        SigmaHat12 = (1.0 / (N - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (N - 1)) * np.dot(H1bar.T, H1bar) + self.r1 * np.identity(f1) # 防止矩阵不可逆
        SigmaHat22 = (1.0 / (N - 1)) * np.dot(H2bar.T, H2bar) + self.r2 * np.identity(f2) # 防止矩阵不可逆

        [D1, V1] = np.linalg.eigh(SigmaHat11) # 计算特征值和特征相向量
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        self.w[0] = np.dot(SigmaHat11RootInv, U[:, 0:self.n_components])
        self.w[1] = np.dot(SigmaHat22RootInv, V[:, 0:self.n_components])
        D = D[0:self.n_components]

    def _get_result(self, x, idx):
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w[idx])
        return result

    def test(self, X1, X2):
        return self._get_result(X1, 0), self._get_result(X2, 1)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-components', type=int, default=10, help='number of cca component')
    parser.add_argument('--missing-rate', type=float, default=0.0, help='view missing rate [default: 0]')
    parser.add_argument('--normalize', action='store_true', default=False, help='whether normalize input features')
    parser.add_argument('--dataset', type=str, default='cmumosi', help='input dataset')
    parser.add_argument('--test_mask', type=str, default=None, help='test under same mask for fair comparision')
    args = parser.parse_args()


    print (f'========= starting ============')
    folder_acc = []
    folder_f1 = []
    folder_save = []
    if args.dataset in ['cmumosi', 'cmumosei']:
        num_folder = 1
    elif args.dataset == 'iemocapfour':
        num_folder = 5
    elif args.dataset == 'iemocapsix':
        num_folder = 5

    for index in range(num_folder):
        print (f'>>>>> Cross-validation: training on the {index+1} folder >>>>>')

        ## read data
        print (f' >>>>> read data >>>>> ')
        X_train, y_train, X_valid, y_valid, X_test, y_test, name_test = read_cmumosi_data(f'../baseline-cpmnet/data/{args.dataset}/{index+1}', args.normalize)
        trainNum = len(y_train)
        validNum = len(y_valid)
        testNum = len(y_test)
        print (f'train number: {trainNum};   valid number: {validNum};   test number: {testNum}')
        X0_train, X1_train, X2_train = X_train
        X0_valid, X1_valid, X2_valid = X_valid
        X0_test, X1_test, X2_test = X_test
        print (f'view 0: {len(X0_train[0])};  view 1: {len(X1_train[0])};  view 2: {len(X2_train[0])}')

        # load random mask
        samplenum = trainNum + validNum + testNum
        Sn = get_sn(3, samplenum, args.missing_rate) # [samplenum, 3] (A, V, L)
        Sn_train = Sn[np.arange(trainNum)]
        Sn_valid = Sn[np.arange(validNum) + trainNum]
        Sn_test =  Sn[np.arange(testNum) + trainNum + validNum]
        ######################################################
        if args.test_mask != None:
            print (f'using predefined mask!!')
            name2mask = np.load(args.test_mask, allow_pickle=True)['name2mask'].tolist()
            Sn_test = []
            for name in name_test:
                mask = name2mask[name] # (A, L, V)
                mask = [mask[0], mask[2], mask[1]] # (A, V, L)
                Sn_test.append(mask)
            Sn_test = np.array(Sn_test) # [sample_num, 3]
        else:
            print (f'using random initialized mask!!')
        ######################################################


        # pad features in masked part
        X0_train, X0_valid, X0_test = padmeanV1(X0_train, Sn_train[:,0], X0_valid, Sn_valid[:,0], X0_test, Sn_test[:,0])
        X1_train, X1_valid, X1_test = padmeanV1(X1_train, Sn_train[:,1], X1_valid, Sn_valid[:,1], X1_test, Sn_test[:,1])
        X2_train, X2_valid, X2_test = padmeanV1(X2_train, Sn_train[:,2], X2_valid, Sn_valid[:,2], X2_test, Sn_test[:,2])


        # CCA feature extraction
        print (f' >>>>> extract features >>>>> ')
        Z_trains = []
        Z_valids = []
        Z_tests = []

        model = CCA(n_components=args.n_components)
        model.fit(X0_train, X1_train)
        Z0_train, Z1_train = model.test(X0_train, X1_train)
        Z0_valid, Z1_valid = model.test(X0_valid, X1_valid)
        Z0_test, Z1_test = model.test(X0_test, X1_test)
        Z_trains.extend([Z0_train, Z1_train])
        Z_valids.extend([Z0_valid, Z1_valid])
        Z_tests.extend([Z0_test, Z1_test])

        model = CCA(n_components=args.n_components)
        model.fit(X0_train, X2_train)
        Z0_train, Z2_train = model.test(X0_train, X2_train)
        Z0_valid, Z2_valid = model.test(X0_valid, X2_valid)
        Z0_test, Z2_test = model.test(X0_test, X2_test)
        Z_trains.extend([Z0_train, Z2_train])
        Z_valids.extend([Z0_valid, Z2_valid])
        Z_tests.extend([Z0_test, Z2_test])

        model = CCA(n_components=args.n_components)
        model.fit(X1_train, X2_train)
        Z1_train, Z2_train = model.test(X1_train, X2_train)
        Z1_valid, Z2_valid = model.test(X1_valid, X2_valid)
        Z1_test, Z2_test = model.test(X1_test, X2_test)
        Z_trains.extend([Z1_train, Z2_train])
        Z_valids.extend([Z1_valid, Z2_valid])
        Z_tests.extend([Z1_test, Z2_test])

        Z_train = np.concatenate(Z_trains, axis=1) # [samplenum, dim]
        Z_valid = np.concatenate(Z_valids, axis=1) # [samplenum, dim]
        Z_test  = np.concatenate(Z_tests, axis=1)  # [samplenum, dim]

        # SVM classify
        print (f' >>>>> training classifier >>>>> ')
        clf = svm.LinearSVC(C=0.01, dual=False)
        clf.fit(Z_train, y_train)
        total_pred = clf.predict(Z_test)
        total_label = y_test
        f1 = f1_score(total_label, total_pred, average='weighted')
        acc = accuracy_score(total_label, total_pred)

        folder_acc.append(acc)
        folder_f1.append(f1)
        folder_save.append({'test_labels': total_label, 'test_preds': total_pred, 'test_hiddens': Z_test, 'test_names': name_test})
        print (f'>>>>> Finish: training on the {index+1} folder >>>>>')

    
    ## save results
    suffix_name = f'{args.dataset}_cca_mask:{args.missing_rate:.1f}'

    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    res_name = f'f1:{mean_f1:2.2%}_acc:{mean_acc:2.2%}'

    save_root = config.MODEL_DIR
    if not os.path.exists(save_root): os.makedirs(save_root)
    save_path = f'{save_root}/{suffix_name}_{res_name}_{time.time()}.npz'
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        folder_save=np.array(folder_save, dtype=object) # save non-structure type
                        )
    print (f' =========== finish =========== ')

