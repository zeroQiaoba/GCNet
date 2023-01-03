import os
import time
import warnings
import numpy as np
from util.util import read_cmumosi_data
from util.get_sn import get_sn
from util.model import CPMNets
import util.classfiy as classfiy
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('../')
import config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=512, help='dimensionality of the latent space data [default: 512]')
    parser.add_argument('--epochs-train', type=int, default=20, metavar='N', help='number of epochs to train [default: 20]')
    parser.add_argument('--epochs-test', type=int, default=100, metavar='N', help='number of epochs to test [default: 50]')
    parser.add_argument('--lamb', type=float, default=10., help='trade off parameter [default: 10]')
    parser.add_argument('--missing-rate', type=float, default=0.0, help='view missing rate [default: 0]')
    parser.add_argument('--normalize', action='store_true', default=False, help='whether normalize input features')
    parser.add_argument('--dataset', type=str, default='cmumosi', help='input dataset')
    parser.add_argument('--test_mask', type=str, default=None, help='test under same mask for fair comparision')
    args = parser.parse_args()

    print (f'========= starting ============')
    folder_f1 = []
    folder_acc = []
    folder_recon = []
    folder_save = []
    if args.dataset in ['cmumosi', 'cmumosei']:
        num_folder = 1
    elif args.dataset == 'iemocapfour':
        num_folder = 5
    elif args.dataset == 'iemocapsix':
        num_folder = 5

    for index in range(num_folder):
        print (f'>>>>> Cross-validation: training on the {index+1} folder >>>>>')

        # read data
        trainData, testData, view_num = read_cmumosi_data(f'data/{args.dataset}/{index+1}', args.normalize)
        outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)] # two view feature dim: [adim, vdim, tdim]

        # set layer size
        layer_size = [[outdim_size[i]] for i in range(view_num)]  # [[adim], [vdim], [tdim]]
        epoch = [args.epochs_train, args.epochs_test]  # [20, 100]
        learning_rate = [0.001, 0.01]

        # Randomly generated missing matrix [under same missing rate]
        Sn = get_sn(view_num, trainData.num_examples + testData.num_examples, args.missing_rate) # [samplenum, 2]
        Sn_train = Sn[np.arange(trainData.num_examples)]
        Sn_test = Sn[np.arange(testData.num_examples) + trainData.num_examples]
        ######################################################
        if args.test_mask != None:
            print (f'using predefined mask!!')
            name2mask = np.load(args.test_mask, allow_pickle=True)['name2mask'].tolist()
            Sn_test = []
            for name in testData.names:
                mask = name2mask[name] # (A, L, V)
                mask = [mask[0], mask[2], mask[1]] # (A, V, L)
                Sn_test.append(mask)
            Sn_test = np.array(Sn_test) # [sample_num, 3]
        else:
            print (f'using random initialized mask!!')
        ######################################################

        # Model building
        model = CPMNets(view_num, trainData.num_examples, testData.num_examples, layer_size, args.lsd_dim, 
                        learning_rate, args.lamb)

        # train: use h -> optimize on y and v [using args.missing_rate]
        model.train(trainData.data, Sn_train, trainData.labels.reshape(trainData.num_examples), epoch[0])
        H_train = model.get_h_train()

        # test: use h -> optimize on v [using all missing rate]
        Imputation_loss = model.test(testData.data, Sn_test, testData.labels.reshape(testData.num_examples), epoch[1])
        H_test = model.get_h_test()

        ## gain prediction results for testData
        total_pred = classfiy.ave(H_train, H_test, trainData.labels)
        total_label = testData.labels
        f1 = f1_score(total_label, total_pred, average='weighted')
        acc = accuracy_score(total_label, total_pred)

        folder_acc.append(acc)
        folder_f1.append(f1)
        folder_recon.append(Imputation_loss / testData.num_examples)
        folder_save.append({'test_labels': total_label, 'test_preds': total_pred, 'test_hiddens': H_test, 'test_names': testData.names})
        print (f'>>>>> Finish: training on the {index+1} folder >>>>>')


    ## save results
    # gain suffix_name
    suffix_name = f'{args.dataset}_cpmnet_mask:{args.missing_rate:.1f}'

    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    mean_recon = np.mean(np.array(folder_recon))
    res_name = f'f1:{mean_f1:2.2%}_acc:{mean_acc:2.2%}_reconloss:{mean_recon:.4f}'

    save_root = config.MODEL_DIR
    if not os.path.exists(save_root): os.makedirs(save_root)
    save_path = f'{save_root}/{suffix_name}_{res_name}_{time.time()}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        folder_save=np.array(folder_save, dtype=object) # save non-structure type
                        )
    print (f' =========== finish =========== ')

