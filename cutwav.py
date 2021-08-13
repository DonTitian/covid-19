#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2021/7/27 下午5:48


#  将原始音频切分成3s
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

file_path = "./DS/test/"
out = './3sDS/test'
path = os.listdir(file_path)

for wave in path:
    wave_path = file_path+wave
    # print(wave_path)
    audio = AudioSegment.from_file(wave_path, "wav")
    size = 3000  # 切割的毫秒数 10s=10000
    chunks = make_chunks(audio, size)  # 将文件切割为10s一块
    name = wave.strip('.wav')  # 去除后缀名，便于保存
    # print(name)
    # name = wave.split('.')[0]
    # print(name)
    for i, chunk in enumerate(chunks):
        chunk_name = out+name+"-{0}.wav".format(i)
        # print(chunk_name)
        # chunk.export(chunk_name, format="wav")

