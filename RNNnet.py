#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2020/10/28 上午11:39


from keras import backend as K
import keras

from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM


# K.set_learning_phase(0)

class rnnNet:

    @staticmethod

    def get_Model(input_shape):
        channel = 1
        num_class = 2

        # Make Network
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 751, 32, 1)

        # Convolution layer (VGG)
        inner = Conv2D(16, (3, 3), padding='same', name='conv0', kernel_initializer='he_normal')(
            inputs)  # (None, 751, 32, 16)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max0')(inner)  # (None,375, 16, 16)

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
            inner)  # (None, 375, 16, 32)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,187, 8, 32)

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
            inner)  # (None, 187, 8, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 1), name='max2')(inner)  # (None, 187, 8, 64)

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
            inner)  # (None, 187, 8, 128)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 45, 4, 128)
        print(inner.shape)

        inner = Reshape(target_shape=(45, 512), name='reshape')(inner)
        inner1 = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        # RNN layer
        lstm_1 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='LSTM1')(inner1)
        lstm_1b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
            inner1)
        reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

        lstm1_merged = add([lstm_1, reversed_lstm_1b])
        lstm1_merged = BatchNormalization()(lstm1_merged)

        lstm_2 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
            lstm1_merged)
        reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)

        lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])
        lstm2_merged = BatchNormalization()(lstm2_merged)
        # print(lstm2_merged.shape)
        lstm2_merged = Flatten()(lstm2_merged)

        # transforms RNN output to character activations:
        net = Dense(2, activation='softmax')(lstm2_merged)
        model = Model(input=inputs, output=net)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        return model