#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2020/10/27 下午2:22


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization


class Net:
    channel = 1
    num_class = 2
    @staticmethod
    def get_model(channel, num_classes):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=(1673, 64, channel)))
        model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), use_bias=True, padding="same"))
        # model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), use_bias=True, padding="same"))
        model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), use_bias=True, padding="same"))
        # model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), use_bias=True, padding="same"))
        model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        #
        model.add(Conv2D(128, (3, 3), use_bias=True, padding="same"))
        # model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), use_bias=True, padding="same"))
        model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), use_bias=True, padding="same"))
        # model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      metrics=['accuracy'])
        model.summary()

        return model