# -*- coding: utf-8 -*-

import keras.backend as K
from keras import models, layers, applications


def build_trip_model(img_h, img_w, n_channel, embedding_size):
    """"""
    base_model = applications.Xception(include_top=False, weights='imagenet', 
                                    input_shape=(img_h, img_w, n_channel), pooling='avg')
    image_input = base_model.input
    out = layers.Dense(embedding_size, activation='relu')(base_model.output)
    out = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')(out)
    image_embeder = models.Model(image_input, out)

    anchor_input = layers.Input(shape=(img_h, img_w, n_channel))
    positive_input = layers.Input(shape=(img_h, img_w, n_channel))
    negative_input = layers.Input(shape=(img_h, img_w, n_channel))

    anchor_output = image_embeder(anchor_input)
    positive_output = image_embeder(positive_input)
    negative_output = image_embeder(negative_input)
    
    trip_model = models.Model(inputs=[anchor_input, positive_input, negative_input], \
                              outputs=[anchor_output, positive_output, negative_output])
    return trip_model
