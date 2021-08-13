#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2021/7/26 下午3:49
import random
import librosa
import os
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import sys

data_train = "./DS/train/cough/"
data_test = "./DS/test"
data_val = "./DS/val/"

def get_labels(path=data_train):
    labels = os.listdir(path)
    # print("labels", labels)
    label_indices = np.arange(0, len(labels))
    # print("label_indices.shape", label_indices.shape)
    # print("label_indices", label_indices)
    # print("to_categorical(label_indices)", to_categorical(label_indices))
    # print("\n")
    return labels, label_indices, to_categorical(label_indices)

def save_data_to_array(path=data_train):
    labels, _, _ = get_labels(path)
    # wavfiles = []
    for label in labels:
        mel_vectors = []
        wavfiles = []
        for wavfile in os.listdir(path + '/' + label + '/'):
            wavfiles.append(path + label + '/' + wavfile)
        # print("wavfiles:", wavfiles)
        for wavfile in tqdm(wavfiles, "saving vectors of label - '{}'".format(label)):
            # print(wavfile)
            mel = np.zeros((1673, 64))
            mel_feature = wav2mel(wavfile)[:, :1673]
            # print("train_mel_feature.shape:", mel_feature.shape)
            mel[:, :mel_feature.shape[1]] = mel_feature
            mel_vectors.append(mel)

        mel_vectors = np.stack(mel_vectors)
        np.save('./npy/train_none_' + label + '.npy', mel_vectors)
        np.save('./npy/train_noise_' + label + '.npy', mel_vectors)

def save_val_data_to_array(path=data_val):
    labels, _, _ = get_labels(path)
    # wavfiles = []
    for label in labels:
        mel_vectors = []
        wavfiles = []
        for wavfile in os.listdir(path + '/' + label + '/'):
            wavfiles.append(path + label + '/' + wavfile)
        # print("wavfiles:", wavfiles)
        for wavfile in tqdm(wavfiles, "saving val vectors of label - '{}'".format(label)):
            mel = np.zeros((1673, 64))
            mel_feature = wav2mel(wavfile)[:, :1673]
            # print("train_mel_feature.shape:", mel_feature.shape)
            mel[:, :mel_feature.shape[1]] = mel_feature
            mel_vectors.append(mel)

        mel_vectors = np.stack(mel_vectors)
        np.save('./npy/val_' + label + '.npy', mel_vectors)

def save_test_data_to_array(path=data_test):
    mel_vectors = []
    wavfiles = []
    for wavfile in os.listdir(path + '/'):
        print(wavfile)
        wavfiles.append(path + '/' + wavfile)
    for wavfile in tqdm(wavfiles, "saving vectors of label - '{}'".format('test')):
        mel = np.zeros((1673, 64))
        mel_feature = wav2mel(wavfile)[:, :1673]
        # print("test_mel_feature.shape:", mel_feature.shape)
        mel[:, :mel_feature.shape[1]] = mel_feature
        mel_vectors.append(mel)

    mel_vectors = np.stack(mel_vectors)
    # np.save('./npy/test.npy', mel_vectors)
    np.save('./npy/val_test.npy', mel_vectors)

def wav2mel(file_path):
    wave, sr = librosa.load(file_path, sr=48000)
    # sr = 48000
    # #归一化
    # wave = wave / max(abs(wave))
    # normalize
    normalize = get_normalize(wave)
    # wave = wave / normalize
    wave = normalize

    mel_spec = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=64, n_fft=512, hop_length=512, power=2.0)
    mel_spec = 20.0 / 2.0 * np.log10(mel_spec + sys.float_info.epsilon)
    # if mel_spec.shape[1] > 180:
    #     print(mel_spec.shape)
    #     print(file_path)
    # pad
    # X = np.zeros((180, 32), dtype=np.float)
    M = np.asarray(mel_spec.T)
    print(M.shape)

    X = np.pad(M, ((((1673 - M.shape[0]) // 2), (1673 - M.shape[0] - ((1673 - M.shape[0]) // 2))),
                   (((64 - M.shape[1]) // 2), (64 - M.shape[1] - ((64 - M.shape[1]) // 2)))),
               'constant', constant_values=(0, 0))
    return X

def get_normalize(data, head_room=0.005):
    mean_value = np.mean(data)
    data = data - mean_value

    max_value = max(abs(data)) + head_room
    return data / max_value

def get_data_to_train():
    labels, indices, _ = get_labels(data_train)
    X = np.load('./npy/train_none_Negative.npy')
    print(X.shape)
    y = np.zeros(X.shape[0])

    x = np.load('./npy/train_none_Positive.npy')
    print("positive.shape", x.shape)  # (160, 180, 32)
    x_fre = np.load("./npy/positive_none_fre_mask.npy")
    print("fre_mask.shape:", x_fre.shape)
    # time的npy
    x_time = np.load("./npy/positive_none_time_mask.npy")
    print("time_mask.shape:", x_time.shape)
    x_noise = np.load('./npy/train_noise_positive.npy')
    frenoise = np.load("./npy/positive_frenoise.npy")  # (160, 180, 32)
    print("frenoise.shape:", frenoise.shape)
    timenoise = np.load("./npy/positive_timenoise.npy")  # (160, 180, 32)
    print("timenoise.shape:", timenoise.shape)
    fre_time_noise = np.load("./npy/positive_fretimenoise.npy")  # (160, 180, 32)
    print("fretimenoise.shape:", fre_time_noise.shape)
    all = np.load("./npy/positive_all.npy")  # (160, 180, 32)
    print("all.shape:", all.shape)
    x1 = np.stack((x, x_noise, x_fre, x_time, frenoise, timenoise, fre_time_noise, all))
    # x1 = np.stack((x, x_fre, x_time))
    x1 = x1.reshape((-1, 1673, 64))
    # 168 = x * 8
    # can not change to 851
    print("x1.shape", x1.shape)

    # X = np.stack((X, x1))
    X = np.append(X, x1, axis=0)
    # can not change to 851
    y = np.append(y, np.full(x1.shape[0], fill_value=1))
    X = X.reshape((-1, 1673, 64))
    print("X.shape", X.shape)
    print("y.shape:", y.shape)
    print(y)
    return X, y

def get_data_to_train1():
    labels, indices, _ = get_labels(data_train)
    X = np.load('./npy/train_noise_Negative.npy')
    y = np.zeros(X.shape[0])
    x = np.load('./npy/train_noise_Positive.npy')
    # print("x.shape", x.shape)
    # print("X.shape", X.shape)
    #因为形状不同，暂时改用其他方式
    # X = np.stack((X, x))
    X = np.append(X,x,axis=0)
    # print("x append shape;", X.shape)
    y = np.append(y, np.full(x.shape[0], fill_value=1))
    X = X.reshape((-1, 1673, 64))
    #训练集
    # print("X.shape", X.shape)
    # print("y.shape:", y.shape)
    return X, y

def get_val_data_to_train(split_ratio=0.8, random_state=42):
    labels, indices, _ = get_labels(data_train)
    X = np.load('./npy/val_Negative.npy')
    y = np.zeros(X.shape[0])
    x = np.load('./npy/val_Positive.npy')
    # print("x.shape", x.shape)
    # X = np.stack((X, x))
    X = np.append(X,x,axis=0)
    # print("x append shape;", X.shape)
    y = np.append(y, np.full(x.shape[0], fill_value=1))
    X = X.reshape((-1, 1673, 64))
    #验证集
    # print("X.shape", X.shape)
    # print("y.shape:", y.shape)
    return X, y

#data augment
def TimeMasking(data, T=5, num_masks=5):
    num_mel_channels = data.shape[0]

    for i in range(0, num_masks):
           t = random.randrange(0, T)
           t_zero = random.randrange(0, num_mel_channels - t)

           # avoids randrange error if values are equal and range is empty
           if (t_zero == t_zero + t): return data

           mask_end = random.randrange(t_zero, t_zero + t)
           data[:, t_zero:mask_end] = 0
    return data

def get_time_mask():
    mel_vectors = []
    path = r"./DS/train/cough/Positive"
    files = os.listdir(path)
    files = [path + "/" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        FileName = files[i]
        print("add time_mask File Name is ", FileName)
        data, fs = librosa.load(files[i], sr=48000, mono=True)
        # normalize
        normalize = get_normalize(data)
        # data = data / normalize
        data =  normalize

        mel_spec = librosa.feature.melspectrogram(data, sr=fs, n_mels=64, n_fft=512, hop_length=512, power=2.0)
        mel_spec = 20.0 / 2.0 * np.log10(mel_spec + sys.float_info.epsilon)
        # pad
        # X = np.zeros((180, 32), dtype=np.float)
        M = np.asarray(mel_spec.T)
        X = np.pad(M, ((((1673 - M.shape[0]) // 2), (1673 - M.shape[0] - ((1673 - M.shape[0]) // 2))),
                       (((64 - M.shape[1]) // 2), (64 - M.shape[1] - ((64 - M.shape[1]) // 2)))),
                   'constant', constant_values=(0, 0))
        data_noise = TimeMasking(X)
        print("data_noise:", data_noise.shape)
        mel = np.zeros((1673, 64))
        mel_feature = data_noise[:, :1673]
        mel[:, :mel_feature.shape[1]] = mel_feature
        mel_vectors.append(mel)

    np.save("./npy/positive_none_time_mask.npy", mel_vectors)
    print('time_mask.npy run over！')

def FrequencyMasking(data, F=5, num_masks=2):
    num_mel_channels = data.shape[1]

    for i in range(0, num_masks):
           f = random.randrange(0, F)
           f_zero = random.randrange(0, num_mel_channels - f)

           # avoids randrange error if values are equal and range is empty
           if (f_zero == f_zero + f): return data

           mask_end = random.randrange(f_zero, f_zero + f)
           data[f_zero:mask_end] = 0
    return data

def get_fre_mask():
    mel_vectors = []
    path = r"./DS/train/cough/Positive"
    files = os.listdir(path)
    files = [path + "/" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        FileName = files[i]
        print("add frequency_mask File Name is ", FileName)
        data, fs = librosa.load(files[i], sr=48000, mono=True)
        # normalize
        normalize = get_normalize(data)
        data = data / normalize

        mel_spec = librosa.feature.melspectrogram(data, sr=fs, n_mels=64, n_fft=512, hop_length=512, power=2.0)
        mel_spec = 20.0 / 2.0 * np.log10(mel_spec + sys.float_info.epsilon)
        # pad
        # X = np.zeros((180, 32), dtype=np.float)
        M = np.asarray(mel_spec.T)
        X = np.pad(M, ((((1673 - M.shape[0])//2), (1673 - M.shape[0] - ((1673 - M.shape[0])//2))),
                   (((64 - M.shape[1])//2), (64 - M.shape[1] - ((64 - M.shape[1])//2)))),
                   'constant', constant_values=(0, 0))
        data_noise = FrequencyMasking(X)
        print("data_noise:", data_noise.shape)
        mel = np.zeros((1673, 64))
        mel_feature = data_noise[:, :1673]
        mel[:, :mel_feature.shape[1]] = mel_feature
        mel_vectors.append(mel)

    np.save("./npy/positive_none_fre_mask.npy", mel_vectors)
    print('frequency_mask.npy run over！')

def get_fre_time_noise():
    mel_vectors = []
    path = r"./DS/train/noise/Positive"
    files = os.listdir(path)
    files = [path + "/" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        FileName = files[i]
        print("add fre_time_mask to noise File Name is ", FileName)
        data, fs = librosa.load(files[i], sr=48000, mono=True)
        # normalize
        normalize = get_normalize(data)
        data = data / normalize

        mel_spec = librosa.feature.melspectrogram(data, sr=fs, n_mels=64, n_fft=512, hop_length=512, power=2.0)
        mel_spec = 20.0 / 2.0 * np.log10(mel_spec + sys.float_info.epsilon)
        # pad
        # X = np.zeros((180, 32), dtype=np.float)
        M = np.asarray(mel_spec.T)
        X = np.pad(M, ((((1673 - M.shape[0]) // 2), (1673 - M.shape[0] - ((1673 - M.shape[0]) // 2))),
                       (((64 - M.shape[1]) // 2), (64 - M.shape[1] - ((64 - M.shape[1]) // 2)))),
                   'constant', constant_values=(0, 0))
        data_noise = FrequencyMasking(X)
        data_noise = TimeMasking(data_noise)
        print("data_noise:", data_noise.shape)
        mel = np.zeros((1673, 64))
        mel_feature = data_noise[:, :1673]
        mel[:, :mel_feature.shape[1]] = mel_feature
        mel_vectors.append(mel)
    np.save("./npy/positive_fretimenoise.npy", mel_vectors)
    print('fre_time_noise.npy run over！')

def get_all_mask():
    mel_vectors = []
    path = r"./DS/train/noise/Positive"
    files = os.listdir(path)
    files = [path + "/" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        FileName = files[i]
        print("add fre_time_mask to noise File Name is ", FileName)
        data, fs = librosa.load(files[i], sr=48000, mono=True)
        # normalize
        normalize = get_normalize(data)
        data = normalize

        mel_spec = librosa.feature.melspectrogram(data, sr=fs, n_mels=64, n_fft=512, hop_length=512, power=2.0)
        mel_spec = 20.0 / 2.0 * np.log10(mel_spec + sys.float_info.epsilon)
        # pad
        # X = np.zeros((180, 32), dtype=np.float)
        M = np.asarray(mel_spec.T)
        X = np.pad(M, ((((1673 - M.shape[0]) // 2), (1673 - M.shape[0] - ((1673 - M.shape[0]) // 2))),
                       (((64 - M.shape[1]) // 2), (64 - M.shape[1] - ((64 - M.shape[1]) // 2)))),
                   'constant', constant_values=(0, 0))
        data_noise = FrequencyMasking(X)
        data_noise = TimeMasking(data_noise)
        print("data_noise:", data_noise.shape)
        mel = np.zeros((1673, 64))
        mel_feature = data_noise[:, :1673]
        mel[:, :mel_feature.shape[1]] = mel_feature
        mel_vectors.append(mel)

    np.save("./npy/positive_all.npy", mel_vectors)
    print('all.npy run over！')

def get_timenoise():
    mel_vectors = []
    path = r"./DS/train/noise/Positive"
    files = os.listdir(path)
    files = [path + "/" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        FileName = files[i]
        print("add time_mask to noise File Name is ", FileName)
        data, fs = librosa.load(files[i], sr=48000, mono=True)
        # normalize
        normalize = get_normalize(data)
        data = normalize

        mel_spec = librosa.feature.melspectrogram(data, sr=fs, n_mels=64, n_fft=512, hop_length=512, power=2.0)
        mel_spec = 20.0 / 2.0 * np.log10(mel_spec + sys.float_info.epsilon)
        # pad
        # X = np.zeros((180, 32), dtype=np.float)
        M = np.asarray(mel_spec.T)
        X = np.pad(M, ((((1673 - M.shape[0]) // 2), (1673 - M.shape[0] - ((1673 - M.shape[0]) // 2))),
                       (((64 - M.shape[1]) // 2), (64 - M.shape[1] - ((64 - M.shape[1]) // 2)))),
                   'constant', constant_values=(0, 0))
        data_noise = FrequencyMasking(X)
        print("data_noise:", data_noise.shape)
        mel = np.zeros((1673, 64))
        mel_feature = data_noise[:, :1673]
        mel[:, :mel_feature.shape[1]] = mel_feature
        mel_vectors.append(mel)

    np.save("./npy/positive_timenoise.npy", mel_vectors)
    print('time_noise.npy run over！')

def get_frenoise():
    mel_vectors = []
    path = r"./DS/train/noise/Positive"
    files = os.listdir(path)
    files = [path + "/" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        FileName = files[i]
        print("add frequency_mask to noise File Name is ", FileName)
        data, fs = librosa.load(files[i], sr=48000, mono=True)
        # normalize
        normalize = get_normalize(data)
        # data = data / normalize
        data = normalize

        mel_spec = librosa.feature.melspectrogram(data, sr=fs, n_mels=64, n_fft=512, hop_length=512, power=2.0)
        mel_spec = 20.0 / 2.0 * np.log10(mel_spec + sys.float_info.epsilon)
        # pad
        # X = np.zeros((180, 32), dtype=np.float)
        M = np.asarray(mel_spec.T)
        X = np.pad(M, ((((1673 - M.shape[0]) // 2), (1673 - M.shape[0] - ((1673 - M.shape[0]) // 2))),
                       (((64 - M.shape[1]) // 2), (64 - M.shape[1] - ((64 - M.shape[1]) // 2)))),
                   'constant', constant_values=(0, 0))
        data_noise = FrequencyMasking(X)
        print("data_noise:", data_noise.shape)
        mel = np.zeros((1673, 64))
        mel_feature = data_noise[:, :1673]
        mel[:, :mel_feature.shape[1]] = mel_feature
        mel_vectors.append(mel)

    np.save("./npy/positive_frenoise.npy", mel_vectors)
    print('fre_noise.npy run over！')

# def findMaxShape(path=data_train):
#     labels, _, _ = get_labels(path)
#     # print(path)
#     # wavfiles = []
#     for label in labels:
#         mel_vectors = []
#         maxNum = 0
#         # print(path + '/' + label + '/')
#         for wavfile in os.listdir(path + '/' + label + '/'):
#             maxNum = wav2mel(wavfile) if maxNum<wav2mel(wavfile) else maxNum
#     return maxNum

# if __name__ == '__main__':
    # result=findMaxShape()
    # print("maxNum：",result)
    # save_data_to_array()
    # save_val_data_to_array()
    # save_test_data_to_array()
    # get_fre_mask()
    # get_time_mask()
    # get_frenoise()
    # get_timenoise()
    # get_fre_time_noise()
    # get_all_mask()
    # wav2mel('./DS/train/cough/Negative/421_Negative_male_26_cough.wav')