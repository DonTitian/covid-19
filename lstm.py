#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2020/11/9 下午1:27


from keras import backend as K
import keras

from keras.layers import Input, Dense
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from Attention import *


class lstm:

    @staticmethod

    def get_Model(input_shape):
        channel = 1
        num_class = 2

        # Make Network
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 751, 32, 1)

        inner = Reshape(target_shape=(1673, 64), name='reshape')(inputs)
        inner1 = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        # RNN layer
        lstm_1 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='LSTM1')(inner1)
        lstm_1b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
            inner)
        reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

        lstm1_merged = add([lstm_1, reversed_lstm_1b])
        lstm1_merged = BatchNormalization()(lstm1_merged)

        lstm_2 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
            lstm1_merged)
        reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)

        lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])
        lstm2_merged = BatchNormalization()(lstm2_merged)

        lstm_3 = LSTM(64, return_sequences=True, kernel_initializer='he_normal', name='lstm3')(lstm2_merged)
        lstm_3b = LSTM(64, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm3_b')(
            lstm2_merged)
        reversed_lstm_3b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_3b)

        lstm3_merged = concatenate([lstm_3, reversed_lstm_3b])
        lstm3_merged = BatchNormalization()(lstm3_merged)
        lstm3_merged = Attention()(lstm3_merged)

        # print(lstm2_merged.shape)
        # lstm3_merged = Flatten()(lstm3_merged)

        # transforms RNN output to character activations:
        net = Dense(2, activation='softmax')(lstm3_merged)
        model = Model(input=inputs, output=net)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      metrics=['accuracy'])
        model.summary()
        return model