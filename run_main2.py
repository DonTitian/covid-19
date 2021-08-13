#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2021/8/5 下午8:21

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
from wav2npy import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

method = "SEnet"
img_path = "./png/" + method + "/"
model_path = "./model/" + method + "/"
csv_path = './save_csv/' + method + '/3x-8-5.csv'


def get_data():
    K.clear_session()
    X, Y = get_data_to_train()
    # 训练
    X_test, y_test = get_val_data_to_train()
    # 验证

    # test_pred = np.zeros((160, 2))
    # 测试
    print("X：", X.shape)
    print("Y:", Y.shape)

    X_train = X
    y_train = Y

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
    plt.savefig(path + str(str(H.history['acc'][-1]) + str(H.history['val_acc'][-1]) + ".png"))


def get_train_model(X_train, y_train, X_test, y_test,method):
    #variables
    channel = 1
    epochs = 10
    batch_size = 16
    verbose = 1
    num_classes = 2

    # 整理形状
    # 4 dimension
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel)

    # expand dimension
    # X_train = np.expand_dims(X_train, axis=3)
    # X_test = np.expand_dims(X_test, axis=3)

    # 3 dimension
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    #
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    # one hot code
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)


    if method=="CNN":
        model = Net.get_model(channel=1, num_classes=2)
    if method=="SEnet":
        # senet
        input_size = (1673, 64, 1)
        model = SEResNeXt(input_size, num_classes=2).model
        model.summary()
        model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
    if method=="LSTM":
        # lstm
        input_size = (1673, 64)
        model = lstm.get_Model(input_size)

    # 设置参数
    for i in range(3):
        my_callbacks = [
            # keras.callbacks.EarlyStopping(patience=20),
            keras.callbacks.ModelCheckpoint(filepath=model_path+'3x-{0}.h5'.format(i), save_best_only=True)]
        H = model.fit(X_train, y_train_hot,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=verbose,
                      validation_data=(X_test, y_test_hot),
                      callbacks=my_callbacks)
        plot_img(H)

# 处理测试集
def get_test_result(method):
    if method=="CNN":
        #cnn
        model = Net.get_model(channel=1, num_classes=2)

    if method=="SEnet":
        #senet
        input_size = (1673, 64, 1)
        model = SEResNeXt(input_size, num_classes=2).model

    if method=="LSTM":
        input_size = (1673, 64)
        model = lstm.get_Model(input_size)

    test_pred = np.zeros((319, 2))
    # model.load_weights('./model/CNN/3x-0.h5')
    model_paths = os.listdir(model_path)
    for path in model_paths:
        model.load_weights(path)
        X_test = np.load('./npy/test.npy')
        test_pred += model.predict(X_test.reshape(319, 1673, 64, 1))
        # test_pred += model.predict(X_test.reshape(160, 180, 32))

    df = pd.DataFrame()
    df['sample_id'] = [wavfile for wavfile in os.listdir(data_test)]
    df['category_id'] = [['Negative', 'Positive'][x] for x in test_pred.argmax(1)]
    # df['category_id'] = [['Positive','Negative' ][x] for x in test_pred.argmax(1)]
    df.to_csv(csv_path, index=None)


if __name__ == '__main__':
    # X_train, y_train, X_test, y_test = get_data()
    # H = get_train_model(X_train, y_train, X_test, y_test,method)
    get_test_result(method)