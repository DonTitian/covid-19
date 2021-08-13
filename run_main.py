#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zyq
# datetime:2021/7/27 下午11:39

import keras
from CNNnet import Net
# from Xceptionnet import XcepNet
# from WaveNet import WaveNet
# from RNNnet import rnnNet
# from RNN_Xcep import rnnNet
# from RNN_Xcep_Attention import rnnNet
# from Dilated import *
from lstm import lstm
from SEnet import *
from resnet import ResnetBuilder

import pytest
from wav2npy import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.80
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 1, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[1], True)

from keras import backend as K

#paths
method = "CNN"
img_path = "./png/" + method + "/"
model_path = "./model/" + method + "/"
csv_path = './save_csv/' + method + '/CNN-8-13-5x50epoch.csv'

#variables
channel = 1
epochs = 50
batch_size = 32
verbose = 1
num_classes = 2

class_weights_dict = {0: 0.7853025936599424, 1: 1.3762626262626263}

def get_data():
    K.clear_session()
    X, Y = get_data_to_train()
    # 训练
    X_test, y_test = get_val_data_to_train()
    # 验证

    # 测试
    # print("X：", X.shape)
    # print("Y:", Y.shape)

    X_train = X
    y_train = Y

    #class weights
    # print("unique:",np.unique(y_train))
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(y_train),
    #                                                   y_train)
    # class_weights_dict = dict(enumerate(class_weights))
    # print('class weight:',class_weights_dict)

    # shuffle train
    result = list(zip(X_train, y_train))  # 打乱的索引序列
    np.random.shuffle(result)
    X_train, y_train = zip(*result)
    # print("y_train:", y_train)

    # shuffle test
    result1 = list(zip(X_test, y_test))  # 打乱的索引序列
    np.random.shuffle(result1)
    X_test, y_test = zip(*result1)
    # print("y_test", y_test)

    # 整理形状
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # if method=="Resnet":
    #     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel)
    #     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel )

    # expand dimension
    # if method=="SEnet":
    #     X_train = np.expand_dims(X_train, axis=3)
    #     X_test = np.expand_dims(X_test, axis=3)
    #
    # # 4 dimension
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel)

    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)

    # one hot code
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def plot_img(H,path = img_path):
    # matplotlib
    print(H.history.keys())
    plt.style.use("ggplot")
    plt.figure()
    N = len(H.history['val_loss'])
    plt.subplot(311)
    plt.plot(np.arange(0, N), H.history['loss'], label="train_loss")
    plt.plot(np.arange(0, N), H.history['val_loss'], label="val_loss")
    plt.title("train loss and val loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend(loc="lower left")
    # 这是分割线
    plt.subplot(313)
    plt.plot(np.arange(0, N), H.history['acc'], label="train_acc")
    plt.plot(np.arange(0, N), H.history['val_acc'], label="val_acc")
    plt.title("train accuracy and val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.legend(loc="lower left")
    # plt.savefig(path + str(str(H.history['acc'][-1]) + str(H.history['val_acc'][-1]) + ".png"))
    plt.savefig(path + str(str(H.history['acc'][-1]) + str(H.history['val_acc'][-1]) + ".png"))


def choose_model(method):
    if method=="CNN":
        model = Net.get_model(channel=1, num_classes=2)
    if method=="SEnet":
        input_size = (1673, 64, 1)
        model = SEResNeXt(input_size, num_classes=2).model
        model.summary()
        model.compile(loss=keras.losses.categorical_crossentropy,
                          # optimizer=keras.optimizers.Adadelta(),
                          optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                          metrics=['accuracy'])
    if method=="LSTM":
        input_size = (1673, 64)
        model = lstm.get_Model(input_size)
    if method=="Resnet":
        model = ResnetBuilder.build_resnet_34((1, 1673, 64),2)
        model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                          metrics=['accuracy'])
    return model


def get_train_model(X_train, y_train, X_test, y_test):

    #cross validation
    seed = 7
    # kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)

    # Merge inputs and targets
    # inputs = np.concatenate((X_train, X_test), axis=0)
    # targets = np.concatenate((y_train, y_test), axis=0)
    inputs = X_train
    targets = y_train

    for i,(train, test) in enumerate(kfold.split(inputs, targets.argmax(1))):#kfold.split(inputs, targets)
        model = choose_model(method)

        #test
        # print('Train: %s | test: %s' % (train, test), '\n')
        # continue

        #early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        my_callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=model_path + '8-13-{0}.h5'.format(i), save_best_only=True),
            early_stopping
        ]

        #calculate class weight
        # print("unique",np.unique(targets[train]))
        # class_weights = class_weight.compute_class_weight('balanced',
        #                                                   np.unique(targets[train]),
        #                                                   targets[train])
        # class_weights_dict = dict(enumerate(class_weights))
        # print(class_weights_dict)

        # one hot code
        # target_train = targets[train]
        # target_train.reshape(1090,-1)
        # target_train = to_categorical(target_train)
        #
        # target_test = targets[test]
        # target_test.reshape(273, -1)
        # target_test = to_categorical(target_test)

        x_train_kf, x_val_kf = inputs[train], inputs[test]
        y_train_kf, y_val_kf = targets[train], targets[test]

        H = model.fit(x_train_kf, y_train_kf,#X_train,y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=verbose,
                      validation_data=(x_val_kf, y_val_kf),#X_test,y_test
                      callbacks=my_callbacks,
                      class_weight=class_weights_dict)#class_weight=class_weights
        plot_img(H)


def get_train_model_none(X_train, y_train, X_test, y_test):

    #cross validation
    # seed = 7
    # kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    # kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)

    # Merge inputs and targets
    # inputs = np.concatenate((X_train, X_test), axis=0)
    # targets = np.concatenate((y_train, y_test), axis=0)
    # inputs = X_train
    # targets = y_train

    # for i,(train, test) in enumerate(kfold.split(inputs, targets.argmax(1))):#kfold.split(inputs, targets)
    model = choose_model(method)
    for i in range(5):
        #early stopping
        # early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        my_callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=model_path + '8-13-out-{0}.h5'.format(i), save_best_only=True),
            # early_stopping
        ]

        #calculate class weight
        # print("unique",np.unique(targets[train]))
        # class_weights = class_weight.compute_class_weight('balanced',
        #                                                   np.unique(targets[train]),
        #                                                   targets[train])
        # class_weights_dict = dict(enumerate(class_weights))
        # print(class_weights_dict)

        # one hot code
        # target_train = targets[train]
        # target_train.reshape(1090,-1)
        # target_train = to_categorical(target_train)
        #
        # target_test = targets[test]
        # target_test.reshape(273, -1)
        # target_test = to_categorical(target_test)

        # x_train_kf, x_val_kf = inputs[train], inputs[test]
        # y_train_kf, y_val_kf = targets[train], targets[test]

        H = model.fit(X_train,y_train,#X_train,y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=verbose,
                      validation_data=(X_test,y_test),#X_test,y_test
                      callbacks=my_callbacks,
                      class_weight=class_weights_dict)#class_weight=class_weights
        plot_img(H)

#validation result
def get_val_result():
    model = choose_model(method)

    test_pred = np.zeros((188, 2))
    # model_paths = os.listdir('./best_code/CNN/model/')
    # this_path = './best_code/CNN/model/'
    # for i in range(len(model_paths)):
    #     model_paths[i] = this_path + model_paths[i]
    # print(model_paths)
    model_paths = os.listdir(model_path)
    for i in range(len(model_paths)):
        model_paths[i] = model_path + model_paths[i]
    print(model_paths)

    for path in model_paths:
        model.load_weights(path)
        # X_test,_ = get_val_data_to_train()
        # X_test = np.asarray(X_test)
        X_test = np.load('./npy/val_test.npy')
        # X_test = np.asarray(X_test)
        print(X_test.shape)
        test_pred += model.predict(X_test.reshape(188, 1673, 64, 1))
        # print(test_pred)

    df = pd.DataFrame()
    df['sample_id'] = [wavfile for wavfile in os.listdir(data_test)]
    df['category_id'] = [['Negative', 'Positive'][x] for x in test_pred.argmax(1)]
    df['real_id'] = [wavfile.split('_')[1] for wavfile in os.listdir(data_test)]
    df.to_csv(csv_path, index=None)


# 处理测试集
def get_test_result():
    model = choose_model(method)

    test_pred = np.zeros((319, 2))
    # model.load_weights('./model/CNN/3x-0.h5')
    model_paths = os.listdir(model_path)
    for i in range(len(model_paths)):
        model_paths[i] = model_path + model_paths[i]
    print(model_paths)

    for path in model_paths:
        model.load_weights(path)
        X_test = np.load('./npy/test.npy')
        # print((X_test.shape))
        test_pred += model.predict(X_test.reshape(-1, 1673, 64, 1))
        # print(test_pred)
        # test_pred += model.predict(X_test.reshape(319, 1673, 64, 1))
        # test_pred += model.predict(X_test.reshape(319, 1673, 64))
        # test_pred += model.predict(X_test.reshape(160, 180, 32))

    df = pd.DataFrame()
    df['sample_id'] = [wavfile for wavfile in os.listdir(data_test)]
    df['category_id'] = [['Negative', 'Positive'][x] for x in test_pred.argmax(1)]
    df.to_csv(csv_path, index=None)


if __name__ == '__main__':
    # X_train, y_train, X_val, y_val= get_data()
    # H = get_train_model(X_train, y_train, X_val, y_val)
    # H = get_train_model_none(X_train, y_train, X_val, y_val)
    get_test_result()
    # get_val_result()