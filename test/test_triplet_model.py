# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import numpy as np
from keras.utils import plot_model
from model import build_trip_model

if __name__ == '__main__':
    img_h, img_w = 300, 300
    n_channel = 3
    embeding_size = 128
    trip_model = build_trip_model(img_h, img_w, n_channel, embeding_size)
    plot_model(trip_model, to_file='trip_mode.png', show_layer_names=True, show_shapes=True)