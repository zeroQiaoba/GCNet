from __future__ import print_function
from __future__ import division

import cv2
import tempfile
import os
import pickle
import random

import numpy as np

from .cifar import CifarDataSet, CifarDataProvider



def augment_image(image, pad):
  """Perform zero padding, randomly crop image to original size,
  maybe mirror horizontally"""
  init_shape = image.shape
  img_size = init_shape[0]
  new_shape = [init_shape[0] + pad * 2,
         init_shape[1] + pad * 2,
         init_shape[2]]
  zeros_padded = np.zeros(new_shape)
  zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
  # randomly crop to original size
  init_x = np.random.randint(0, pad * 2)
  init_y = np.random.randint(0, pad * 2)
  cropped = zeros_padded[
    init_x: init_x + init_shape[0],
    init_y: init_y + init_shape[1],
    :]
  # randomly flip
  flip = random.getrandbits(1)
  if flip:
    cropped = cropped[:, ::-1, :]
  # randomly rotation
  angle = np.random.randint(-15, 16)
  rot_mat = cv2.getRotationMatrix2D((img_size, img_size), angle, 1.)
  cropped = cv2.warpAffine(cropped, rot_mat, (img_size, img_size))
  if len(cropped.shape) == 2:
    cropped = np.expand_dims(cropped, 2)
  return cropped


def augment_all_images(initial_images, pad):
  new_images = np.zeros(initial_images.shape)
  for i in range(initial_images.shape[0]):
    new_images[i] = augment_image(initial_images[i], pad=pad)
  return new_images


class FERPlusDataSet(CifarDataSet):


  # def start_new_epoch(self):
  #   self._batch_counter = 0
  #   if self.shuffle_every_epoch:
  #     images, labels = self.shuffle_images_and_labels(
  #       self.images, self.labels)
  #   else:
  #     images, labels = self.images, self.labels
  #   if self.augmentation:
  #     images = augment_all_images(images, pad=8)
  #   self.epoch_images = images
  #   self.epoch_labels = labels

  def next_batch(self, batch_size):
    start = self._batch_counter * batch_size
    end = (self._batch_counter + 1) * batch_size
    self._batch_counter += 1
    images_slice = self.images[self.random_idxs[start: end]]
    labels_slice = self.labels[self.random_idxs[start: end]]
    if images_slice.shape[0] != batch_size:
      self.start_new_epoch()
      return self.next_batch(batch_size)
    else:
      if self.augmentation:
        images_slice = augment_all_images(images_slice, pad=8)
      return images_slice, labels_slice


class FERPlusDataProvider(CifarDataProvider):

  def __init__(self, data_dir, shuffle=None, normalization=None, 
    data_augmentation=False, **kwargs):
    """
    Args:
      data_dir: `str`
      shuffle: `str` or None
        None: no any shuffling
        once_prior_train: shuffle train data only once prior train
        every_epoch: shuffle train data prior every epoch
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
        by_chanels: substract mean of every chanel and divide each
          chanel data by it's standart deviation
    """
    self.data_augmentation = data_augmentation
    print('data augmentation', data_augmentation)

    # load data
    dataset = {}
    for setname in ['trn', 'val', 'tst']:
      dataset[setname] = {}
      dataset[setname]['img'] = np.expand_dims(
        np.load(os.path.join(data_dir, '%s_img.npy'%setname)), 3)
      dataset[setname]['target'] = np.load(os.path.join(data_dir, '%s_target.npy'%setname))
    self._n_classes = len(np.unique(np.argmax(dataset['trn']['target'], 1)))

    # add train and validations datasets
    self.train = FERPlusDataSet(
      images=dataset['trn']['img'], labels=dataset['trn']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)

    self.validation = FERPlusDataSet(
      images=dataset['val']['img'], labels=dataset['val']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)
    
    # add test set
    self.test = FERPlusDataSet(
      images=dataset['tst']['img'], labels=dataset['tst']['target'],
      shuffle=None, n_classes=self.n_classes,
      normalization=normalization,
      augmentation=False)

  @property
  def data_shape(self):
    return (64, 64, 1)

class AVECDataProvider(FERPlusDataProvider):
  def __init__(self, data_dir, target_idxs=[0, 1], shuffle=None, normalization=None, 
    data_augmentation=False, **kwargs):
    """
    Args:
      data_dir: `str`
      shuffle: `str` or None
        None: no any shuffling
        once_prior_train: shuffle train data only once prior train
        every_epoch: shuffle train data prior every epoch
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
        by_chanels: substract mean of every chanel and divide each
          chanel data by it's standart deviation
    """
    self.data_augmentation = data_augmentation
    print('data augmentation', data_augmentation)
    self.target_idxs = np.array(target_idxs)


    # load data
    dataset = {}
    for setname in ['trn', 'val']:#, 'tst']:
      dataset[setname] = {}
      dataset[setname]['img'] = np.expand_dims(
        np.load(os.path.join(data_dir, '%s_img.npy'%setname)), 3).astype(np.float32)
      dataset[setname]['target'] = np.load(
        os.path.join(data_dir, '%s_target.npy'%setname))[:, self.target_idxs]
    self._n_classes = len(target_idxs)

    # add train and validations datasets
    self.train = FERPlusDataSet(
      images=dataset['trn']['img'], labels=dataset['trn']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)

    self.validation = FERPlusDataSet(
      images=dataset['val']['img'], labels=dataset['val']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)
    
    # add test set
    # self.test = FERPlusDataSet(
    #   images=dataset['val']['img'], labels=dataset['val']['target'],
    #   shuffle=None, n_classes=self.n_classes,
    #   normalization=normalization,
    #   augmentation=False)

class MUSEDataProvider(FERPlusDataProvider):
  def __init__(self, data_dir, target_idxs=[0, 1], shuffle=None, normalization=None, 
    data_augmentation=False, **kwargs):
    """
    Args:
      data_dir: `str`
      shuffle: `str` or None
        None: no any shuffling
        once_prior_train: shuffle train data only once prior train
        every_epoch: shuffle train data prior every epoch
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
        by_chanels: substract mean of every chanel and divide each
          chanel data by it's standart deviation
    """
    self.data_augmentation = data_augmentation
    print('data augmentation', data_augmentation)
    self.target_idxs = np.array(target_idxs)


    # load data
    dataset = {}
    for setname in ['trn', 'val']:#, 'tst']:
      dataset[setname] = {}
      dataset[setname]['img'] = np.expand_dims(
        np.load(os.path.join(data_dir, '%s_img.npy'%setname)), 3).astype(np.float32)
      dataset[setname]['target'] = np.load(
        os.path.join(data_dir, '%s_target.npy'%setname))[:, self.target_idxs]
    self._n_classes = len(target_idxs)

    # add train and validations datasets
    self.train = FERPlusDataSet(
      images=dataset['trn']['img'], labels=dataset['trn']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)

    self.validation = FERPlusDataSet(
      images=dataset['val']['img'], labels=dataset['val']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)
    
    images_all = np.concatenate([dataset['trn']['img'], dataset['val']['img']])
    labels_all = np.concatenate([dataset['trn']['target'], dataset['val']['target']])
    self.all = FERPlusDataSet(
      images=images_all, labels=labels_all,
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)

class VGGFACE2DataProvieder(FERPlusDataProvider):
  def __init__(self, data_dir, target_idxs=[0], shuffle=None, normalization=None, 
    data_augmentation=False, **kwargs):
    """
    Args:
      data_dir: `str`
      shuffle: `str` or None
        None: no any shuffling
        once_prior_train: shuffle train data only once prior train
        every_epoch: shuffle train data prior every epoch
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
        by_chanels: substract mean of every chanel and divide each
          chanel data by it's standart deviation
    """
    self.data_augmentation = data_augmentation
    print('data augmentation', data_augmentation)
    self.target_idxs = np.array(target_idxs)

    # load data
    dataset = {}
    for setname in ['trn', 'val']:#, 'tst']:
      dataset[setname] = {}
      dataset[setname]['img'] = np.expand_dims(
        np.load(os.path.join(data_dir, '%s_img.npy'%setname)), 3).astype(np.float32)
      dataset[setname]['target'] = np.load(
        os.path.join(data_dir, '%s_target.npy'%setname))[:, self.target_idxs]
    self._n_classes = len(target_idxs)

    # add train and validations datasets
    self.train = FERPlusDataSet(
      images=dataset['trn']['img'], labels=dataset['trn']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)

    self.validation = FERPlusDataSet(
      images=dataset['val']['img'], labels=dataset['val']['target'],
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)
    
    images_all = np.concatenate([dataset['trn']['img'], dataset['val']['img']])
    labels_all = np.concatenate([dataset['trn']['target'], dataset['val']['target']])
    self.all = FERPlusDataSet(
      images=images_all, labels=labels_all,
      n_classes=self.n_classes, shuffle=shuffle,
      normalization=normalization,
      augmentation=self.data_augmentation)

if __name__ == '__main__':
  # some sanity checks for Cifar data providers
  import matplotlib.pyplot as plt
  import sys

  data_dir = sys.argv[1]

  # plot some CIFAR10 images with classes
  def plot_images_labels(images, labels, axes, main_label, classes):
    plt.text(0, 1.5, main_label, ha='center', va='top',
         transform=axes[len(axes) // 2].transAxes)
    for image, label, axe in zip(images, labels, axes):
      axe.imshow(image[:, :, 0], cmap='gray')
      axe.set_title(classes[np.argmax(label)])
      axe.set_axis_off()

  # fer_idx_to_class = ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']
  # fer_provider = FERPlusDataProvider(data_dir)
  # # assert fer_provider._n_classes == 8
  # # assert fer_provider.train.labels.shape[-1] == 8
  # # assert len(fer_provider.train.labels.shape) == 2
  # # assert fer_provider.train.images.shape[0] == 28452
  # # assert fer_provider.validation.images.shape[0] == 3569
  # # assert fer_provider.test.images.shape[0] == 3517

  # # test shuffling
  # fer_provider_not_shuffled = FERPlusDataProvider(data_dir, shuffle=None)
  # fer_provider_shuffled = FERPlusDataProvider(data_dir, shuffle='once_prior_train')
  # assert not np.all(
  #   fer_provider_not_shuffled.train.images != fer_provider_shuffled.train.images)
  # assert np.all(
  #   fer_provider_not_shuffled.test.images == fer_provider_shuffled.test.images)

  # n_plots = 10
  # fig, axes = plt.subplots(nrows=4, ncols=n_plots)
  # plot_images_labels(
  #   fer_provider_not_shuffled.train.images[:n_plots],
  #   fer_provider_not_shuffled.train.labels[:n_plots],
  #   axes[0],
  #   'Original dataset',
  #   fer_idx_to_class)
  # dataset = FERPlusDataProvider(data_dir, normalization='divide_256')
  # plot_images_labels(
  #   dataset.train.images[:n_plots],
  #   dataset.train.labels[:n_plots],
  #   axes[1],
  #   'Original dataset normalized dividing by 256',
  #   fer_idx_to_class)
  # dataset = FERPlusDataProvider(data_dir, normalization='by_chanels', data_augmentation=True)
  # dataset = AVECDataProvider(data_dir, normalization='by_chanels', data_augmentation=False)
  dataset = MUSEDataProvider(data_dir, normalization='by_chanels', data_augmentation=False)
  # dataset = VGGFACE2DataProvieder(data_dir, normalization='by_chanels', data_augmentation=False)
  print(dataset.train.images_means, dataset.train.images_stds)
  print(dataset.validation.images_means, dataset.validation.images_stds)
  print(dataset.all.images_means, dataset.all.images_stds)
  # print(dataset.test.images_means, dataset.test.images_stds)
  # plot_images_labels(
  #   dataset.train.epoch_images[:n_plots],
  #   dataset.train.epoch_labels[:n_plots],
  #   axes[2],
  #   'Original dataset normalized by mean/std at every channel',
  #   fer_idx_to_class)
  # plot_images_labels(
  #   fer_provider_shuffled.train.images[:n_plots],
  #   fer_provider_shuffled.train.labels[:n_plots],
  #   axes[3],
  #   'Shuffled dataset',
  #   fer_idx_to_class)
  # plt.show()