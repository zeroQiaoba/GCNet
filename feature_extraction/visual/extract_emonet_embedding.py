# *_*coding:utf-8 *_*
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np

from emonet.models.emonet import EmoNet
from dataset import FaceDatasetForEmoNet
from util import write_feature_to_csv, get_vids, write_feature_to_npy
from emonet.data_augmentation import DataAugmentor

# import config
import sys
sys.path.append('../../')
import config

def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, names in tqdm(data_loader):
            images = images.cuda()
            embedding = model(images, return_embedding=True)
            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps



def main(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    print(f'==> Extracting emonet embedding...')
    # in: face dir
    face_dir = config.PATH_TO_RAW_FACE[params.dataset]
    # out: feature csv dir
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], 'emonet')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # load model
    model = EmoNet().cuda()
    # model = torch.nn.DataParallel(model).cuda()
    checkpoint_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'emonet/emonet_8.pth')
    checkpoint = torch.load(checkpoint_file)
    pre_trained_dict = {k.replace('module.', ''): v for k,v in checkpoint.items()}
    model.load_state_dict(pre_trained_dict)

    # transform
    augmentor = DataAugmentor(256, 256)
    transform = transforms.Compose([transforms.ToTensor()])

    # extract embedding video by video
    vids = get_vids(face_dir)
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # forward
        dataset = FaceDatasetForEmoNet(vid, face_dir, transform=transform, augmentor=augmentor)
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            features, timestamps = [], []
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
            features, timestamps = extract(data_loader, model)
            
        # write
        # write_feature_to_csv(features, timestamps, save_dir, vid, feature_dim=feature_dim)
        write_feature_to_npy(features, timestamps, save_dir, vid)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=str, default='5', help='gpu id')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    params = parser.parse_args()

    main(params)