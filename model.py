# -*- coding: utf-8 -*-

import keras.backend as K
from keras import models, layers, applications

from config import img_h, img_w, n_channels, embedding_size


def build_trip_model():
    """"""
    base_model = applications.Xception(include_top=False, weights=None, 
                                    input_shape=(img_h, img_w, n_channels), pooling='avg')

    image_input = base_model.input
    out = layers.Dense(embedding_size, activation='relu')(base_model.output)
    out = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')(out)
    image_embeder = models.Model(image_input, out)

    anchor_input = layers.Input(shape=(img_h, img_w, n_channels))
    positive_input = layers.Input(shape=(img_h, img_w, n_channels))
    negative_input = layers.Input(shape=(img_h, img_w, n_channels))

    anchor_output = image_embeder(anchor_input)
    positive_output = image_embeder(positive_input)
    negative_output = image_embeder(negative_input)

    output = layers.concatenate(inputs=[anchor_output, positive_output, negative_output], name='concat')
    
    trip_model = models.Model(inputs=[anchor_input, positive_input, negative_input], \
                              outputs=[output])
    return trip_model