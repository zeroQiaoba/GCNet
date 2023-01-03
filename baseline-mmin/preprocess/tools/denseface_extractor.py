
import os, glob
import cv2
import numpy as np
import tensorflow as tf
import collections

from preprocess.tools.denseface.vision_network.models.dense_net import DenseNet

class DensefaceExtractor(object):
    def __init__(self, restore_path=None, mean=131.0754, std=47.858177, device=0, smooth=False):
        """ extract densenet feature
            Parameters:
            ------------------------
            model: model class returned by function 'load_model'
        """
        if restore_path is None:
            restore_path = '/data2/zjm/tools/FER_models/denseface/DenseNet-BC_growth-rate12_depth100_FERPlus/model/epoch-200'
        self.model = self.load_model(restore_path)
        self.mean = mean
        self.std = std
        self.previous_img = None        # smooth 的情况下, 如果没有人脸则用上一张人脸填充
        self.previous_img_path = None
        self.smooth = smooth
        self.dim = 342                  # returned feature dim
        self.device = device
    
    def load_model(self, restore_path):
        print("Initialize the model..")
        # fake data_provider
        growth_rate = 12
        img_size = 64
        depth = 100
        total_blocks = 3
        reduction = 0.5
        keep_prob = 1.0
        bc_mode = True
        model_path = restore_path
        dataset = 'FER+'
        num_class = 8

        DataProvider = collections.namedtuple('DataProvider', ['data_shape', 'n_classes'])
        data_provider = DataProvider(data_shape=(img_size, img_size, 1), n_classes=num_class)
        model = DenseNet(data_provider=data_provider, growth_rate=growth_rate, depth=depth,
                        total_blocks=total_blocks, keep_prob=keep_prob, reduction=reduction,
                        bc_mode=bc_mode, dataset=dataset)

        model.saver.restore(model.sess, model_path)
        print("Successfully load model from model path: {}".format(model_path))
        return model
    
    def __call__(self, img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if not isinstance(img, np.ndarray):
                print(f'Warning: Error in {img_path}')
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            if self.smooth:
                self.previous_img = img
                self.previous_img_path = img_path

        elif self.smooth and self.previous_img is not None:
            # print('Path {} does not exists. Use previous img: {}'.format(img_path, self.previous_img_path))
            img = self.previous_img
        
        else:
            feat = np.zeros([1, self.dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
            return feat
        
        img = (img - self.mean) / self.std
        img = np.expand_dims(img, -1) # channel = 1
        img = np.expand_dims(img, 0) # batch_size=1
        with tf.device('/gpu:{}'.format(self.device)):
            feed_dict = {
                self.model.images: img,
                self.model.is_training: False
            }

            # emo index
            # fer_idx_to_class = ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']

            ft, soft_label = \
                self.model.sess.run([self.model.end_points['fc'], 
                                     self.model.end_points['preds']], feed_dict=feed_dict)
        return ft, soft_label