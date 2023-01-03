# *_*coding:utf-8 *_*
from __future__ import division

import os
import time
import six
import sys
from tqdm import tqdm
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data
from os.path import join as pjoin
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import glob
import numbers
from PIL import Image, ImageOps
import random
import copy
# for torch lower version
import torch._utils
from torch.nn import functional as F

from dataset import FaceDataset
from util import write_feature_to_csv, get_vids, write_feature_to_npy

# import config
import sys
sys.path.append('../../')
import config

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3
    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition
    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod


def load_model(model_name, model_dir, pretrained_dir):
    """Load imoprted PyTorch model by name
    Args:
        model_name (str): the name of the model to be loaded
    Return:
        nn.Module: the loaded network
    """
    model_def_path = pjoin(model_dir, model_name + '.py')
    weights_path = pjoin(pretrained_dir, model_name + '.pth')
    # print(os.path.abspath(model_def_path))
    # print(os.path.abspath(weights_path))
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net


def compose_transforms(meta, resize=256, center_crop=True,
                       override_meta_imsize=False):
    """Compose preprocessing transforms for model
    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.
    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `resize`
           to select the image input size, rather than the properties contained
           in meta (this option only applies when center cropping is not used.
    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if center_crop:
        transform_list = [transforms.Resize(resize),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]

    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]:  # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)


def get_feature(model, layer_name, image):
    bs = image.size(0)
    layer = model._modules.get(layer_name)
    if layer_name == 'fc7':
        my_embedding = torch.zeros(bs, 4096)
    elif layer_name == 'fc8':
        my_embedding = torch.zeros(bs, 7)
    elif layer_name == 'pool5' or layer_name == 'pool5_full':
        my_embedding = torch.zeros([bs, 512, 7, 7])
    elif layer_name == 'pool4':
        my_embedding = torch.zeros([bs, 512, 14, 14])
    elif layer_name == 'pool3':
        my_embedding = torch.zeros([bs, 256, 28, 28])
    elif layer_name == 'pool5_7x7_s1':  # available
        my_embedding = torch.zeros([bs, 2048, 1, 1])
    elif layer_name == 'conv5_3_3x3_relu': # available
        my_embedding = torch.zeros([bs, 512, 7, 7])
    else:
        raise Exception(f'Error: not supported layer "{layer_name}".')

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    h = layer.register_forward_hook(copy_data)
    _ = model(image)
    h.remove()
    if layer_name == 'pool5' or layer_name == 'conv5_3_3x3_relu':
        GAP_layer = nn.AvgPool2d(kernel_size=[7, 7], stride=(1, 1))
        my_embedding = GAP_layer(my_embedding)

    my_embedding = F.relu(my_embedding.squeeze())
    if my_embedding.size(0) != bs:
        my_embedding = my_embedding.unsqueeze(0)
    my_embedding = my_embedding.detach().cpu().numpy().tolist()
    return my_embedding


def extract(data_loader, model, layer_name):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for imgs, ids in data_loader:
            imgs = imgs.cuda()
            batch_features = get_feature(model, layer_name, imgs)
            features.extend(batch_features)
            timestamps.extend(ids)
        features, timestamps = np.array(features), np.array(timestamps)
        return features, timestamps


def main(params):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu

    print(f'==> Extracting ferplus embedding...')
    # in: face dir
    face_dir = config.PATH_TO_RAW_FACE[params.dataset]
    # out: feature csv dir
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f"{params.model_name.split('_')[0]}face_{params.layer_name.split('_')[0]}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # load pre-trained model
    pretrained_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'ferplus') # directory of pre-trained models
    model_dir = './pytorch-benchmarks/model' # directory of model definition file
    model = load_model(params.model_name, model_dir, pretrained_dir)
    meta = model.meta
    device = torch.device("cuda")
    model = model.to(device)

    # transform
    transform = compose_transforms(meta)

    # extract embedding video by video
    vids = get_vids(face_dir)
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # forward
        dataset = FaceDataset(vid, face_dir, transform=transform)
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            features, timestamps = [], []
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
            features, timestamps = extract(data_loader, model, params.layer_name)
            # feature_dim = features.shape[1]
        # write
        # face_rate = write_feature_to_csv(features, timestamps, save_dir, vid, feature_dim=feature_dim)
        write_feature_to_npy(features, timestamps, save_dir, vid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--model_name', type=str, default='resnet50_ferplus_dag',
                        choices=['resnet50_ferplus_dag', 'senet50_ferplus_dag'],
                        help='name of pretrained model')
    parser.add_argument('--layer_name', type=str, default='pool5_7x7_s1', choices=['pool5_7x7_s1', 'conv5_3_3x3_relu'],
                        help='which layer used to extract feature')
    parser.add_argument('--gpu', type=str, default='4', help='gpu id')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    params = parser.parse_args()

    main(params)