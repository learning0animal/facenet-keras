# -*- coding: utf-8 -*-

import keras.backend as K

def triplet_loss(anchor, positive, negative, margin=1.0):
    """ 参考 https://pytorch.org/docs/0.3.1/_modules/torch/nn/functional.html#triplet_margin_loss
    """
    assert anchor.shape == positive.shape
    assert anchor.shape == negative.shape
    assert positive.shape == negative.shape
    assert margin > 0.0
    
    # TODO: 不是特别理解,为什么是先sum后sqrt
    d_p = K.sqrt(K.sum(K.square(anchor - positive), axis=1, keepdims=True))
    d_n = K.sqrt(K.sum(K.square(anchor - negative), axis=1, keepdims=True))
    d_hinge = K.max(d_p - d_n + margin, 0)
    return K.mean(d_hinge)
