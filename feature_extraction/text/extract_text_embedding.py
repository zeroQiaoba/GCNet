# *_*coding:utf-8 *_*
import os
import glob
import math
import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm
import itertools
from transformers import AutoModel, AutoTokenizer # version: 4.5.1, pip install transformers
import re
import argparse
from util import write_feature_to_csv, load_word2vec, load_glove, strip_accent

# import config
import sys
sys.path.append('../../')
import config

"""
supported models for tasks 1-2
"""
# word vectors
GLOVE = 'glove/glove.840B.300d.txt'
WORD2VEC = 'word2vec/GoogleNews-vectors-negative300.bin'
# transformers
# BERT
BERT_BASE = 'bert-base-cased'
BERT_LARGE = 'bert-large-cased'
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_LARGE_UNCASED = 'bert-large-uncased'
# ALBERT
ALBERT_BASE = 'albert-base-v2'
ALBERT_LARGE = 'albert-large-v2'
ALBERT_XXLARGE = 'albert-xxlarge-v2'
# RoBERTa
ROBERTA_BASE = 'roberta-base'
ROBERTA_LARGE = 'roberta-large'
# ELECTRA
ELECTRA_BASE = 'electra-base-discriminator'
ELECTRA_LARGE = 'electra-large-discriminator'
# GPT
GPT = 'openai-gpt'
GPT2 = 'gpt2'
GPT2_MEDIUM = 'gpt2-medium'
GPT2_LARGE = 'gpt2-large'
GPT_NEO = 'gpt-neo-1.3B'
# XLNet
XLNET_BASE = 'xlnet-base-cased'
XLNET_LARGE = 'xlnet-large-cased'
# T5
T5_BASE = 't5-base'
T5_LARGE = 't5-large'
# DeBERTa
DEBERTA_BASE = 'deberta-base'
DEBERTA_LARGE = 'deberta-large'
DEBERTA_XLARGE = 'deberta-v2-xlarge'
DEBERTA_XXLARGE = 'deberta-v2-xxlarge'

"""
supported models for tasks 3-4 (Pretrained on German corpus)
"""
BERT_GERMAN_CASED = 'bert-base-german-cased'
BERT_GERMAN_DBMDZ_CASED = 'bert-base-german-dbmdz-cased'
BERT_GERMAN_DBMDZ_UNCASED = 'bert-base-german-dbmdz-uncased'



def extract_bert_embedding(model_name, trans_dir, save_dir, dir_name=None, layer_ids=None, combine_type='mean',
                           batch_size=256, gpu=6, overwrite=False):
    """
    :param model_name: which pre-trained model
    :param trans_dir: the directory of transcriptions
    :param save_dir: the root directory used to store feature csv files
    :param dir_name: name of directory in which feature csv files are stored, if specified
    :param layer_ids: hidden states of selected layers will be summed as the word embedding. For example, "-1" denotes output of the last layer.
    :param combine_type: how to combine sub-word embeddings to obtain the whole word embedding.
    :return:
    """
    print('='*30 + f' Extracting "{model_name}" ' + '='*30)
    start_time = time.time()

    # layer ids
    if layer_ids is None:
        layer_ids = [-4, -3, -2, -1]
    else:
        assert isinstance(layer_ids, list)

    # save_dir
    if dir_name is None: # use model_name for naming if dir_name is None
        dir_name = model_name if len(layer_ids) == 1 else f'{model_name}-{len(layer_ids)}'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite csv out dir "{dir_name}"!')
    else:
        raise Exception(f'==> Error: csv out dir "{dir_name}" already exists, set overwrite=TRUE if needed!')

    # load model and tokenizer: offline mode (load cached files)
    print('Loading pre-trained tokenizer and model...')
    model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False) # do not use fast version of Tokenizer (because of failure of RoBERTa )
    if model_name == GPT_NEO:
        model = AutoModel.from_pretrained(model_dir, from_tf=True)
    else:
        model = AutoModel.from_pretrained(model_dir)
    device = torch.device(f'cuda:{gpu}')
    model.to(device)
    model.eval()

    # iterate videos
    vids = sorted(os.listdir(trans_dir), key=lambda x: int(x))
    for idx, vid in enumerate(tqdm(vids), 1):
        print(f'Processing "{vid}" ({idx}/{len(vids)})...')
        # extract sentences and timestamps from raw files
        sentences, sentence = [], []
        timestamps = []
        v_dir = os.path.join(trans_dir, vid)
        trans_files = glob.glob(os.path.join(v_dir, '*.csv'))
        trans_files.sort(key=lambda x: int(os.path.basename(os.path.splitext(x)[0]).split('_')[1]))
        if 'german' in model_name: # special treatment for German transcriptions
            for file in trans_files:
                segment_df = pd.read_csv(file)
                if len(segment_df) > 0:
                    sentence = []
                    for _, row in segment_df.iterrows():
                        word, s_t, e_t = row['word'], row['start'], row['end']
                        if 'uncased' in model_name:
                            word = strip_accent(word.lower()) # to lower case and strip accent
                        sentence.append(word)
                        timestamps.append((s_t, e_t))
                    sentences.append(sentence)
        else: # for English transcriptions
            # concat df
            segment_dfs = []
            for file in trans_files:
                segment_df = pd.read_csv(file)
                segment_dfs.append(segment_df)
            df = pd.concat(segment_dfs)
            for _, row in df.iterrows():
                word = row['word']
                if word in ['.', '!', '?']:
                    if sentence != []:
                        sentences.append(sentence)
                        sentence = []
                else:
                    s_t, e_t = row['start'], row['end']
                    if (e_t - s_t) > 1: # denotes word (for punctuation, the interval is always 1)
                        word_cleaned = re.sub(r'[^a-zA-Z0-9,.\'!?]+', '', word) # remove special character
                        if 'uncased' in model_name or 'albert' in model_name or 'electra' in model_name: # Note: to lower case if needed, different from previous version, 2021/05/04
                            word_cleaned = word_cleaned.lower()
                        if word_cleaned:
                            sentence.append(word_cleaned)
                            timestamps.append((s_t, e_t))
            if sentence != []: # some files do not end with '.', '!', '?'
                sentences.append(sentence)
        words = list(itertools.chain(*sentences))
        assert len(words) == len(timestamps), print(sentence)
        ### Note: For BERT model, they take sentence as input

        # extract embedding from sentences
        embeddings = []
        n_batches = math.ceil(len(sentences) / batch_size)
        for i in range(n_batches):
            s_idx, e_idx = i * batch_size, min((i+1) * batch_size, len(sentences))
            batch_sentences = sentences[s_idx:e_idx]
            inputs = tokenizer(batch_sentences, padding=True, is_split_into_words=True, return_tensors='pt') # generate input_ids, attention_mask
            input_ids = inputs['input_ids']
            inputs = inputs.to(device)
            with torch.no_grad():
                # outputs = model(**inputs, output_hidden_states=True)[2] # for old version 3.0.2
                outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0) # sum => [batch, T, D=768]
                outputs = outputs.cpu().numpy() # (B, T, D)
                lens = torch.sum(inputs['attention_mask'], dim=1) # each sample's lens
                real_batch_size = outputs.shape[0]
                for i in range(real_batch_size): # for each sample
                    if 'xlnet' in model_name:
                        input_id = input_ids[i, -lens[i]:-2] # (T,) Note, for left pading and skipping ('<sep>', '<cls>')
                        output = outputs[i, -lens[i]:-2]  # (T, D)
                    else: # lens[0] = 21
                        input_id = input_ids[i, 1:(lens[i] - 1)] # (T,) Note, 1:(lens[i] - 1) for skipping [CLS] and [SEP]
                        output = outputs[i, 1:(lens[i] - 1)] # (T, D)
                    sentence = batch_sentences[i]
                    # for debug
                    # print('='*30)
                    # print(tokenizer.convert_ids_to_tokens(input_ids[i]))
                    n_tokens, n_words = output.shape[0], len(sentence)
                    if n_tokens == n_words: # sub-word is word
                        sentence_embedding = list(output)
                        embeddings.extend(sentence_embedding)
                    else: # align sub-word to word
                        sentence_embedding = []
                        pointer = 0
                        word, word_embedding = '', []
                        sentence_words, sentence_tokens = [], []
                        for j, token_id in enumerate(input_id):
                            token = tokenizer.convert_ids_to_tokens([token_id])[0]
                            sentence_tokens.append(token)
                            token_embedding = output[j]
                            current_word = sentence[pointer]
                            token = token.replace('▁', '') #  for albert-like (also, xlnet) model (ex, hello: '▁hello')
                            token = token.replace('Ġ', '') #  for roberta-like (also, gpt) model (ex, was: 'Ġwas')
                            if token == current_word or token == '[UNK]': # take care of unknown words
                                sentence_embedding.append(token_embedding)
                                sentence_words.append(token)
                                pointer += 1
                            else:
                                word_embedding.append(token_embedding)
                                token = token.replace('##', '') # for bert model
                                word = word + token
                                if  word == current_word: # ended token
                                    if combine_type == 'sum':
                                        word_embedding = np.sum(np.row_stack(word_embedding), axis=0) # sum sub-word emebddings
                                    elif combine_type == 'mean':
                                        word_embedding = np.mean(np.row_stack(word_embedding), axis=0) # average sub-word emebddings
                                    elif combine_type == 'last':
                                        word_embedding = word_embedding[-1] # take the last sub-word emebdding
                                    else:
                                        raise Exception('Error: not supported type to combine subword embedding.')
                                    sentence_embedding.append(word_embedding)
                                    sentence_words.append(word)
                                    word, word_embedding = '', []
                                    pointer += 1
                        assert len(sentence) == len(sentence_embedding), \
                            print(f'==>len(sentence): {len(sentence)}, len(embedding): {len(sentence_embedding)}\ntokens:{sentence_tokens}\nwords:{sentence_words}\nsentence:{sentence}')
                        embeddings.extend(sentence_embedding)
        assert len(embeddings) == len(timestamps)
        # align with label timestamp and write csv file
        csv_file = os.path.join(save_dir, f'{vid}.csv')
        log_file = os.path.join('./log', dir_name, f'{vid}.csv')
        write_feature_to_csv(embeddings, timestamps, words, csv_file, log_file=log_file, embedding_dim=model.config.hidden_size)
    end_time = time.time()
    print(f'Total {len(vids)} files done! Time used ({model_name}): {end_time - start_time:.1f}s.')



## Target: extract word embedding from trans_dir
def extract_word_embedding(model_name, trans_dir, save_dir, dir_name=None, overwrite=False):
    assert model_name in [WORD2VEC, GLOVE], 'Error: not supported word embedding file!'
    start_time = time.time()

    # trans_concat_dir & save_dir
    trans_concat_dir = os.path.join(trans_dir, '../transcription_concat')
    if not os.path.exists(trans_concat_dir):
        os.makedirs(trans_concat_dir)

    if dir_name is None:
        dir_name = 'word2vec' if model_name == WORD2VEC else 'glove'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite csv out dir "{dir_name}"!')
    else:
        raise Exception(f'==> Error: csv out dir "{dir_name}" already exists, set overwrite=TRUE if needed!')

    # # load embeddings dict
    # if model_name == GLOVE:
    #     embedding_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, GLOVE)
    #     EMBEDDINGS, EMBEDDING_DIM = load_glove(embedding_file)
    # else:
    #     embedding_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, WORD2VEC)
    #     EMBEDDINGS, EMBEDDING_DIM = load_word2vec(embedding_file)
    # print(f'Use word embedding file "{embedding_file}".')

    # iterate videos
    vids = sorted(os.listdir(trans_dir), key=lambda x: int(x))
    for idx, vid in enumerate(tqdm(vids), 1):
        print(f'Processing {vid} ({idx}/{len(vids)})...')
        # extract sentences and timestamps from raw files
        v_dir = os.path.join(trans_dir, vid)
        trans_files = glob.glob(os.path.join(v_dir, '*.csv'))
        trans_files.sort(key=lambda x: int(os.path.basename(os.path.splitext(x)[0]).split('_')[1]))
        segment_dfs = []
        for file in trans_files:
            segment_df = pd.read_csv(file)
            segment_dfs.append(segment_df)
        df = pd.concat(segment_dfs) ## concat all segment_dfs => [1058, 4]
        trans_concat_file = os.path.join(trans_concat_dir, f'{vid}.csv')
        df.to_csv(trans_concat_file) # save segment_dfs to "trans_concat_file"
        words, timestamps = [], []
        for _, row in df.iterrows(): # read each row
            word = row['word']
            s_t, e_t = row['start'], row['end']
            # if word not in [',', '.', '!', '?'] or (e_t - s_t) > 1:
            if word not in [',', '.', '!', '?']:
                assert (e_t - s_t) > 1
                words.append(word.lower()) # to lower case
                timestamps.append((s_t, e_t))
            else:
                assert (e_t - s_t) == 1, print(word)

        # extract embedding
        ## then read all words (remove .,!?) and timestamps
        embeddings = []
        not_matched = []
        for word in words:
            if word in EMBEDDINGS:
                embeddings.append(EMBEDDINGS[word])
            else:
                not_matched.append(word)
                embeddings.append(np.zeros((EMBEDDING_DIM,)))
        print(f'Total {len(not_matched)} non-matched words: {" ".join(not_matched)}.')
        # align with label timestamp and write csv file
        csv_file = os.path.join(save_dir, f'{vid}.csv')
        log_file = os.path.join('./log', model_name, f'{vid}.csv')
        write_feature_to_csv(embeddings, timestamps, words, csv_file, log_file=log_file)
    end_time = time.time()
    print(f'Time used ({model_name}): {end_time - start_time:.1f}s.')



def main(model_name, trans_dir, save_dir, layer_ids, overwrite, gpu, batch_size):
    if model_name in [WORD2VEC, GLOVE]:
        extract_word_embedding(model_name=model_name, 
                               trans_dir=trans_dir, 
                               save_dir=save_dir, 
                               overwrite=overwrite)
    else: # transformers
        extract_bert_embedding(model_name=model_name,
                               trans_dir=trans_dir,
                               save_dir=save_dir,
                               layer_ids=layer_ids,
                               batch_size=batch_size,
                               gpu=gpu,
                               overwrite=overwrite)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run.')
    # choose one from supported models. Note: some transformers listed above are not available due to various errors!
    parser.add_argument('--model_name', type=str, default=WORD2VEC,
                        choices=[WORD2VEC, BERT_BASE, BERT_LARGE, 
                        BERT_BASE_UNCASED, BERT_LARGE_UNCASED, ALBERT_BASE,
                        ALBERT_LARGE, ALBERT_XXLARGE, ROBERTA_BASE, ROBERTA_LARGE,
                        XLNET_BASE, XLNET_LARGE, T5_BASE, T5_LARGE, DEBERTA_BASE, 
                        DEBERTA_LARGE, DEBERTA_XLARGE, DEBERTA_XXLARGE],
                        help='name of pretrained model')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    args = parser.parse_args()
    
    trans_dir = config.PATH_TO_TRANSCRIPTIONS # directory of transcriptions
    save_dir = config.PATH_TO_FEATURES # directory used to store features
    layer_ids = [-4, -3, -2, -1] # hidden states of selected layers will be used to obtain the embedding, only for transformers

    main(args.model_name, trans_dir, save_dir, layer_ids=layer_ids, overwrite=args.overwrite, gpu=args.gpu, batch_size=16)

