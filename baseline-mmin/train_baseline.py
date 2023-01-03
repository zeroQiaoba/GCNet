import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test'):
    model.eval()

    total_pred = []
    total_label = []
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test() # forward and gain results
        pred = model.pred.detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)
    total_pred = np.concatenate(total_pred)   # [sample_num, ]
    total_label = np.concatenate(total_label) # [sample_num, ]

    dataset = opt.dataset_mode.split('_')[0]
    if dataset in ['cmumosi','cmumosei']:
        non_zeros = np.array([i for i, e in enumerate(total_label) if e != 0]) # remove 0, and remove mask
        acc = accuracy_score((total_label[non_zeros] > 0), (total_pred[non_zeros] > 0))
        f1 = f1_score((total_label[non_zeros] > 0), (total_pred[non_zeros] > 0), average='weighted')
    elif dataset in ['iemocapfour', 'iemocapsix']:
        total_pred = np.argmax(total_pred, 1)
        acc = round(accuracy_score(total_label, total_pred), 2)
        f1 = round(f1_score(total_label, total_pred, average='weighted'), 2)

    # save test results
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    model.train()
    return acc, f1


def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))



if __name__ == '__main__':
    opt = TrainOptions().parse()                        # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo)) # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.mkdir(logger_path)

    result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'), total_cv=12) # init result recoreder
    suffix = '_'.join([opt.model, opt.dataset_mode])    # get logger suffix: utt_fusion_multimodal
    logger = get_logger(logger_path, suffix)            # get logger

    dataset = opt.dataset_mode.split('_')[0]
    if dataset in ['cmumosi', 'cmumosei']:
        assert opt.output_dim == 1
        num_folder = 1
    elif dataset == 'iemocapfour':
        assert opt.output_dim == 4
        num_folder = 5
    elif dataset == 'iemocapsix':
        assert opt.output_dim == 6
        num_folder = 5

    folder_acc = []
    folder_f1 = []
    folder_save = []

    for index in range(num_folder):
        print (f'>>>>> Cross-validation: training on the {index+1} folder >>>>>')
        opt.cvNo = index + 1

        ## reading data
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])  

        dataset_size = len(dataset)    # get the number of images in the dataset.
        logger.info('The number of training samples = %d' % dataset_size) # sample number: 5531
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        total_iters = 0                # the total number of training iterations
        best_eval_uar = 0              # record the best eval UAR
        best_epoch_acc, best_epoch_f1 = 0, 0
        best_eval_epoch = -1           # record the best eval epoch

        ## epoch: start from opt.epoch_count
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()   # timer for computation per iteration
                total_iters += 1                # opt.batch_size
                model.set_input(data)           # unpack data from dataset and apply preprocessing
                model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights
                    
                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                model.save_networks('latest')
                model.save_networks(epoch)

            logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.

            # eval val set
            acc, f1 = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f f1 %.4f' % (epoch, opt.niter + opt.niter_decay, acc, f1))

            if f1 > best_epoch_f1:
                best_eval_epoch = epoch
                best_epoch_acc = acc
                best_epoch_f1 = f1
        
        # test on best epoch
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        acc, f1 = eval(model, tst_dataset, is_save=True, phase='test')
        folder_acc.append(acc)
        folder_f1.append(f1)
        clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)

        print (f'>>>>> Finish: training on the {index+1} folder >>>>>')
        
