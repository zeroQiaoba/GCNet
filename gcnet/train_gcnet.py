import os
import time
import glob
import math
import pickle
import random
import argparse
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.append('../')
import config

from model import GraphModel
from dataloader_iemocap import IEMOCAPDataset
from dataloader_cmumosi import CMUMOSIDataset
from loss import MaskedCELoss, MaskedMSELoss, MaskedReconLoss

def get_loaders(audio_root, text_root, video_root, num_folder, dataset, batch_size, num_workers, seed):

    ###########################################################################
    ###########################################################################
    if dataset in ['CMUMOSI', 'CMUMOSEI']:

        dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum+valNum))
        test_idxs = list(range(trainNum+valNum, trainNum+valNum+testNum))

        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=SubsetRandomSampler(train_idxs),
                                  collate_fn=dataset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=False)
        val_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(val_idxs),
                                collate_fn=dataset.collate_fn,
                                num_workers=num_workers,
                                pin_memory=False)
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=SubsetRandomSampler(test_idxs),
                                 collate_fn=dataset.collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=False)

        train_loaders = [train_loader]
        val_loaders = [val_loader]
        test_loaders = [test_loader]

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, val_loaders, test_loaders, adim, tdim, vdim


    ###########################################################################
    ###########################################################################
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']: ## five folder cross-validation, each fold contains (train, test)

        dataset = IEMOCAPDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        ## gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
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

        ## gain train and test loaders
        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                      collate_fn=dataset.collate_fn,
                                      num_workers=num_workers,
                                      pin_memory=False)
            test_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=SubsetRandomSampler(test_idxs),
                                     collate_fn=dataset.collate_fn,
                                     num_workers=num_workers,
                                     pin_memory=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, test_loaders, adim, tdim, vdim


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    graph_h = args.hidden // 2
    model = GraphModel(args.base_model,
                       adim, tdim, vdim, D_e, graph_h,
                       n_speakers=args.n_speakers,
                       window_past=args.windowp,
                       window_future=args.windowf,
                       n_classes=args.n_classes,
                       dropout=args.dropout,
                       time_attn=args.time_attn,
                       no_cuda=args.no_cuda)
    print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print ('Graph NN with', args.base_model, 'as base model.')
    return model


## gain input features: ?*[seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = [] 
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, featdim=adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1) # [batch, seqlen] -> [seqlen, batch]
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask==0, feat1, feat2) # -> [seqlen, batch, featdim]
    input_features.append(select_feat) # 1 * [seqlen, batch, dim]
    return input_features


## follow cpm-net's masking manner
def random_mask(view_num, input_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """

    assert missing_rate is not None
    one_rate = 1 - missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(input_len, view_num)) # [samplenum, viewnum] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:input_len, :]
    return matrix


def train_or_eval_model(args, model, reg_loss, cls_loss, rec_loss, dataloader, 
                        mask_rate=None, optimizer=None, train=False):
    preds, masks, labels, vidnames = [], [], [], []
    savepreds, savelabels, savespeakers, savehiddens, savefmask = [], [], [], [], []
    losses, losses1, losses2 = [], [], []

    dataset = args.dataset
    reccls_flag = args.reccls_flag
    lower_bound = args.lower_bound
    cuda = torch.cuda.is_available() and not args.no_cuda

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train: optimizer.zero_grad()
        
        ## read dataloader
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        """
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        vidnames += data[-1]
        adim = audio_host.size(2)
        tdim = text_host.size(2)
        vdim = visual_host.size(2)

        ## using cmp-net masking manner [at least one view exists]
        """
        ?_?_mask: [seqlen, batch, dim]   => gain mask
        masked_?_?: [seqlen, batch, dim] => masked features

        # if audio_feature is None: audio_feature = text_feature
        # if text_feature is None: text_feature = audio_feature
        # if video_feature is None: video_feature = text_feature
        # mask sure, same mask for same features [include padded features]
        """
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)
        ## host mask [!!use original audio_feature!!]
        view_num = 3
        matrix = random_mask(view_num, seqlen*batch, mask_rate) # [seqlen*batch, view_num]
        audio_host_mask = np.reshape(matrix[:, 0], (seqlen, batch, 1)) 
        text_host_mask = np.reshape(matrix[:, 1], (seqlen, batch, 1))
        visual_host_mask = np.reshape(matrix[:, 2], (seqlen, batch, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask)
        text_host_mask = torch.LongTensor(text_host_mask)
        visual_host_mask = torch.LongTensor(visual_host_mask)

        # guest mask
        view_num = 3
        matrix = random_mask(view_num, seqlen*batch, mask_rate) # [seqlen*batch, view_num]
        audio_guest_mask = np.reshape(matrix[:, 0], (seqlen, batch, 1)) 
        text_guest_mask = np.reshape(matrix[:, 1], (seqlen, batch, 1))
        visual_guest_mask = np.reshape(matrix[:, 2], (seqlen, batch, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask)
        text_guest_mask = torch.LongTensor(text_guest_mask)
        visual_guest_mask = torch.LongTensor(visual_guest_mask)
        if view_num == 2: assert mask_rate <= 0.500001, f'Warning: at least one view exists'
        if view_num == 3: assert mask_rate <= 0.700001, f'Warning: at least one view exists'

        ## lower bound==True => remove missing data
        if not lower_bound:
            masked_audio_host = audio_host * audio_host_mask
            masked_audio_guest = audio_guest * audio_guest_mask
            masked_text_host = text_host * text_host_mask
            masked_text_guest = text_guest * text_guest_mask
            masked_visual_host = visual_host * visual_host_mask
            masked_visual_guest = visual_guest * visual_guest_mask
        else:
            host_mask = torch.logical_and(torch.logical_and(audio_host_mask, text_host_mask), visual_host_mask).int() # [seqlen, bacth, 1]
            masked_audio_host = audio_host * host_mask
            masked_text_host = text_host * host_mask
            masked_visual_host = visual_host * host_mask
            audio_host_mask = host_mask
            text_host_mask = host_mask
            visual_host_mask = host_mask
            guest_mask = torch.logical_and(torch.logical_and(audio_guest_mask, text_guest_mask), visual_guest_mask).int() # [seqlen, bacth, 1]
            masked_audio_guest = audio_guest * guest_mask
            masked_text_guest = text_guest * guest_mask
            masked_visual_guest = visual_guest * guest_mask
            audio_guest_mask = guest_mask
            text_guest_mask = guest_mask
            visual_guest_mask = guest_mask

        ## add cuda for tensor
        if cuda:
            audio_host = audio_host.cuda()
            text_host = text_host.cuda()
            visual_host = visual_host.cuda()
            audio_guest = audio_guest.cuda()
            text_guest = text_guest.cuda()
            visual_guest = visual_guest.cuda()

            masked_audio_host, audio_host_mask = masked_audio_host.cuda(), audio_host_mask.cuda()
            masked_text_host, text_host_mask = masked_text_host.cuda(), text_host_mask.cuda()
            masked_visual_host, visual_host_mask = masked_visual_host.cuda(), visual_host_mask.cuda()
            masked_audio_guest, audio_guest_mask = masked_audio_guest.cuda(), audio_guest_mask.cuda()
            masked_text_guest, text_guest_mask = masked_text_guest.cuda(), text_guest_mask.cuda()
            masked_visual_guest, visual_guest_mask = masked_visual_guest.cuda(), visual_guest_mask.cuda()

            qmask = qmask.cuda()
            umask = umask.cuda()
            label = label.cuda()

        ## [conversation_len1, conversation_len2, ..., conversation_lenN]
        lengths = []
        for j in range(len(umask)):
            length = (umask[j] == 1).nonzero().tolist()[-1][0] + 1 
            lengths.append(length)

        ## generate input_features: ? * [seqlen, batch, dim]
        input_features = generate_inputs(audio_host, text_host, visual_host, \
                                         audio_guest, text_guest, visual_guest, qmask)
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                                audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)

        '''
        # input_features, masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # log_prob: [seqlen, batch, num_classes]
        # input_features_recon # padded, ?*[seqlen, batch, dim]
        '''
        if reccls_flag: # whether use reconstruction features for classification
            _, recon_input_features, _ = model(masked_input_features, qmask, umask, lengths)
            log_prob, _, hidden = model(recon_input_features, qmask, umask, lengths)
        else:
            log_prob, recon_input_features, hidden = model(masked_input_features, qmask, umask, lengths)

        ## gain saved results [utterance-level]
        tempseqlen = np.sum(umask.cpu().data.numpy(), 1) # [batch]
        temphidden = hidden.transpose(0,1).cpu().data.numpy() # [batch, seqlen, featdim]
        temppred = log_prob.transpose(0,1).cpu().data.numpy() # [batch, seqlen, num_classes]
        templabel = label.cpu().data.numpy() # [batch, seqlen]
        tempqmask = qmask.cpu().data.numpy() # [batch, seqlen]
        tempfmask = input_features_mask[0].transpose(0,1).cpu().data.numpy() # [seqlen, batch, 3] -> [batch, seqlen, 3]
        for ii in range(len(tempseqlen)): # utt_number for each conversation
            itemhidden = temphidden[ii][:int(tempseqlen[ii]), :] # [seqlen, featdim]
            itempred   = temppred[ii][:int(tempseqlen[ii]), :]   # [seqlen, num_classes]
            itemfmask  = tempfmask[ii][:int(tempseqlen[ii]), :]  # [seqlen, 3]
            itemlabel  = templabel[ii][:int(tempseqlen[ii])]     # [len, ]
            itemspks   = tempqmask[ii][:int(tempseqlen[ii])]     # [len, ]
            savehiddens.append(itemhidden)
            savepreds.append(itempred)
            savefmask.append(itemfmask)
            savelabels.append(itemlabel)
            savespeakers.append(itemspks)

        ## calculate loss
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size(2)) # [batch*seq_len, n_classes]
        labels_ = label.view(-1) # [batch*seq_len]
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']: loss1 = cls_loss(lp_, labels_, umask)
        if dataset in ['CMUMOSI', 'CMUMOSEI']  : loss1 = reg_loss(lp_, labels_, umask)
        loss2 = rec_loss(recon_input_features, input_features, input_features_mask, umask, adim, tdim, vdim)
        if args.loss_recon: loss = loss1 + loss2
        if not args.loss_recon: loss = loss1
        
        ## save batch results
        # pred_ = torch.argmax(lp_,1) # [batch*seq_len]
        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        losses1.append(loss1.item()*masks[-1].sum())
        losses2.append(loss2.item()*masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_loss1 = round(np.sum(losses1)/np.sum(masks), 4)
        avg_loss2 = round(np.sum(losses2)/np.sum(masks), 4)
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
    elif dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0]) # remove 0, and remove mask
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_loss1 = round(np.sum(losses1)/np.sum(masks), 4)
        avg_loss2 = round(np.sum(losses2)/np.sum(masks), 4)
        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        
    print (f'sample number: {np.sum(masks)}')
    return avg_accuracy, avg_fscore, vidnames, [avg_loss, avg_loss1, avg_loss2], [savepreds, savelabels, savespeakers, savehiddens, savefmask]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    ## Params for model
    parser.add_argument('--base-model', type=str, choices=['LSTM', 'GRU'], help='base recurrent model, must be one of LSTM/GRU')
    parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--windowp', type=int, default=6, help='context window size for constructing edges in graph model for past utterances, -1: fully connect')
    parser.add_argument('--windowf', type=int, default=6, help='context window size for constructing edges in graph model for future utterances, -1: fully connect')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')

    ## Params for training
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--mask-type', type=str, default='constant-0.1', help='mask rate [0~1] for input argumentation: constant-float; linear; convex; concave')
    parser.add_argument('--loss-recon', action='store_true', default=False, help='whether to use reconstrctuion loss')
    parser.add_argument('--reccls-flag', action='store_true', default=False, help='whether to use reconstrctuion features for classification')
    parser.add_argument('--lower-bound', action='store_true', default=False, help='whether remove missing modality in the training process')
    args = parser.parse_args()

    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)


    print (f'====== Reading Data =======')
    audio_feature = args.audio_feature
    text_feature = args.text_feature
    video_feature = args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders( audio_root = audio_root,
                                                                              text_root  = text_root,
                                                                              video_root = video_root,
                                                                              num_folder = args.num_folder,
                                                                              batch_size = args.batch_size,
                                                                              dataset = args.dataset,
                                                                              num_workers = 0,
                                                                              seed = args.seed)
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    
    print (f'====== Training and Evaluation =======')
    folder_acc = []       # save best epoch
    folder_f1 = []        # save best epoch
    folder_recon = []     # save best epoch
    folder_save = []      # save best epoch
    folder_losswhole = [] # save whole epoch
    folder_savewhole = [] # save whole epoch
    for ii in range(args.num_folder):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        val_loader = val_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print (f'Step1: build model (each folder has its own model)')
        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        rec_loss = MaskedReconLoss()
        if cuda:
            model.cuda()
            cls_loss.cuda()
            rec_loss.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        print (f'Step2: training (multiple epoches)')
        all_losses = []
        all_labels = []
        val_fscores = []
        test_fscores, test_accs, test_recon = [], [], []
        for epoch in range(args.epochs):
            assert args.mask_type.startswith('constant'), f'mask_type should be constant-x.x'
            mask_rate = float(args.mask_type.split('-')[-1])
           
            ## training, validation and testing
            train_acc, train_fscore, train_names, train_loss, trainsave = train_or_eval_model(args, model, reg_loss, cls_loss, rec_loss, train_loader, \
                                                                            mask_rate=mask_rate, optimizer=optimizer, train=True)
            val_acc, val_fscore, val_names, val_loss, valsave = train_or_eval_model(args, model, reg_loss, cls_loss, rec_loss, val_loader, \
                                                                            mask_rate=mask_rate, optimizer=None, train=False)
            test_acc, test_fscore, test_names, test_loss, testsave = train_or_eval_model(args, model, reg_loss, cls_loss, rec_loss, test_loader, \
                                                                            mask_rate=mask_rate, optimizer=None, train=False)

            ## save
            val_fscores.append(val_fscore)
            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_recon.append(test_loss[2])
            all_losses.append({'train_loss':train_loss, 'val_loss':val_loss, 'test_loss':test_loss})
            all_labels.append({'test_labels':testsave[1], 'test_preds':testsave[0], 'test_hiddens':testsave[3], 'test_names':test_names, 'test_fmask':testsave[4]})
            print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_loss:{train_loss[0]}; train_loss1:{train_loss[1]}; train_loss2:{train_loss[2]}')

        print (f'Step3: saving and testing on the {ii+1} folder')
        best_index = np.argmax(np.array(val_fscores))
        bestf1 = test_fscores[best_index]
        bestacc = test_accs[best_index]
        bestrecon = test_recon[best_index]
        bestsave = all_labels[best_index]
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_recon.append(bestrecon)
        folder_save.append(bestsave)
        folder_losswhole.append(all_losses)
        assert args.epochs >= 60, f'epoch number should large then 60'
        folder_savewhole.append([best_index, all_labels[10], all_labels[20], all_labels[50], all_labels[best_index]])
        end_time = time.time()
        print (f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')


    print (f'====== Saving =======')
    save_root = config.MODEL_DIR
    if not os.path.exists(save_root): os.makedirs(save_root)
    ## gain suffix_name
    mask_rate = args.mask_type.split('-')[-1]
    suffix_name = f'{args.dataset.lower()}_Graph{args.base_model}_mask:{mask_rate}'
    ## gain feature_name and cls_name
    feature_name = f'{audio_feature};{text_feature};{video_feature}'
    cls_name = f'lossrecon:{args.loss_recon}+lower:{args.lower_bound}+reccls:{args.reccls_flag}'
    ## gain res_name
    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    mean_recon = np.mean(np.array(folder_recon))
    res_name = f'f1:{mean_f1:2.2%}_acc:{mean_acc:2.2%}_reconloss:{mean_recon:.4f}'

    save_path = f'{save_root}/{suffix_name}_features:{feature_name}_classifier:{cls_name}_{res_name}_{time.time()}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        folder_losswhole=np.array(folder_losswhole, dtype=object),
                        folder_savewhole=np.array(folder_savewhole, dtype=object)
                        )