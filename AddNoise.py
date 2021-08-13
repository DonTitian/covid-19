#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2020/10/26 下午3:38


import librosa
import numpy as np
import os

def add_noise(data):
    wn = np.random.normal(0, 0.1, len(data))
    data_noise = np.where(data != 0.0, data.astype('float64') + 0.01 * wn, 0.0).astype(np.float32)
    return data_noise


file_path = './DS/train/cough/Positive/'
out_path = './DS/train/noise/Positive/'
path = os.listdir(file_path)
#
for wave in path:
    wave_path = file_path+wave
    data, fs = librosa.core.load(wave_path, sr=None)
    print(fs)
    data_noise = add_noise(data)
    out_file = out_path + 'noise_' + wave
    librosa.output.write_wav(out_file, data_noise, fs)

print("run over!")
