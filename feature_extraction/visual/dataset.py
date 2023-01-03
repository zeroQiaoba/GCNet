# *_*coding:utf-8 *_*
import os
import glob
from PIL import Image
from skimage import io
import torch.utils.data as data


class FaceDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FaceDataset, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        # if len(frames) == 0:
        # raise ValueError("number of frames of video {} should not be zero.".format(self.vid))
        # frames = sorted(frames, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))
        # frame_ids = [int(os.path.basename(os.path.splitext(file)[0])) for file in frames]

        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        # fid = int(os.path.basename(os.path.splitext(path)[0]))
        name = os.path.basename(path)[:-4]
        return img, name



class FaceDatasetForEmoNet(data.Dataset):
    def __init__(self, vid, face_dir, transform=None, augmentor=None):
        super(FaceDatasetForEmoNet, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.augmentor = augmentor
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        # frames = sorted(frames, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = io.imread(path)
        if self.augmentor is not None:
            img = self.augmentor(img)[0]
        if self.transform is not None:
            img = self.transform(img)
        # fid = int(os.path.basename(os.path.splitext(path)[0]))
        # return img, fid
        name = os.path.basename(path)[:-4]
        return img, name