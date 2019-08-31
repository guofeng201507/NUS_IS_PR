#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:44:44 2019

@author: doppiomovimento
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate


def createDualTwModel():
    shared = Conv2D(32, (3, 3), padding = 'same', activation='relu')

    left_ipt = Input(shape=(32, 32, 3))
    left_x = Conv2D(16, (3, 3), padding='same', activation='relu')(left_ipt)
    left_x = Conv2D(16, (3, 3), padding='same', activation='relu')(left_x)
    left_x = MaxPooling2D(pool_size=(2, 2))(left_x)
    left_x = shared(left_x)
    left_x = MaxPooling2D(pool_size=(2, 2))(left_x)

    right_ipt = Input(shape=(16, 16, 3))
    right_x = Conv2D(16, (3, 3), padding='same', activation='relu')(right_ipt)
    right_x = shared(right_x)
    right_x = MaxPooling2D(pool_size=(2, 2))(right_x)

    shared_48 = Conv2D(48, (3, 3), activation='relu')

    left_x = shared_48(left_x)
    left_x = MaxPooling2D(pool_size=(2, 2))(left_x)

    right_x = shared_48(right_x)
    right_x = MaxPooling2D(pool_size=(2, 2))(right_x)

    shared_64 = Conv2D(48, (3, 3), activation='relu')

    left_x = shared_64(left_x)
    left_x = MaxPooling2D(pool_size=(2, 2))(left_x)

    right_x = shared_64(right_x)
    right_x = MaxPooling2D(pool_size=(2, 2))(right_x)

    x = concatenate([left_x, right_x], axis=-1)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs=[left_ipt, right_ipt], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

modelDualTw = createDualTwModel()
modelDualTw.summary()

from tensorflow.keras.utils import plot_model

plot_model(modelDualTw,
           to_file='DualTw_model.pdf',
           show_shapes=True,
           show_layer_names=False,
           rankdir='TB')
