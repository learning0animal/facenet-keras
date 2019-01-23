# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from itertools import combinations, product, permutations

from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
from keras.applications.inception_resnet_v2 import preprocess_input

from config import batch_size, img_h, img_w, n_channels, embedding_size



class TripDataGenerator(Sequence):
    
    def __init__(self, path):

        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.n_channels = n_channels
        self.embedding_size = embedding_size

        self.path = path
        self.labels = os.listdir(self.path)
        
        self.conbs = list(filter(lambda x: x[0] != x[1], list(product(self.labels, self.labels))))
        np.random.shuffle(self.conbs)
        self.n = len(self.conbs)

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def __getitem__(self, idx):

        idx_i = idx * self.batch_size
        idx_j = min((idx + 1) * self.batch_size, self.n)
        
        # 生成图片集合
        # batch_x = np.zeros((3, self.batch_size, self.img_h, self.img_w, self.n_channels))
        batch_x_fns = []
        for conb in self.conbs[idx_i: idx_j]:
            p_dir = os.path.join(self.path, conb[0])            
            n_dir = os.path.join(self.path, conb[1])
            # 从 p_dir 挑选两张图片,一张作为 anchor 一张作为 positive
            fns = []
            for i in range(2):
                fn = np.random.choice(os.listdir(p_dir))
                fns.append(os.path.join(p_dir, fn))
            # 从 n_dir 挑选一张图片,作为 negative
            for i in range(2, 3):
                fn = np.random.choice(os.listdir(n_dir))
                fns.append(os.path.join(n_dir, fn))
            batch_x_fns.append(fns)

        batch_input = np.array([[preprocess_input(resize(imread(file_name), (self.img_h, self.img_w))) \
                for file_name in con_fns] for con_fns in batch_x_fns])
        
        # (batch_size, 3, img_h, img_w, n_channels)
        # print(batch_input.shape)
        y_true = np.zeros((self.batch_size, 3 * self.embedding_size))
        return [batch_input[:, 0, :], batch_input[:, 1, :], batch_input[:, 2, :]], y_true

    def on_epoch_end(self):
        np.random.shuffle(self.conbs)

        
if __name__ == '__main__':
    gen = TripDataGenerator('./data/train')
    # print(gen.conbs)
    item = gen.__getitem__(0)
    # print(item)
    print(item[0][0].shape)