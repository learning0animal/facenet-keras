# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import numpy as np
from model import triplet_loss

if __name__ == '__main__':
    anchor = np.random.randn(1, 128)
    positive = np.random.randn(1, 128)
    negative = np.random.randn(1, 128)
    print(triplet_loss(anchor, positive, negative))