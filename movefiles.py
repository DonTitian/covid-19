#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:ttt
# datetime:2021/7/29 下午5:52
import os
import shutil
import  librosa
from pydub import AudioSegment
from pydub.utils import make_chunks

path = "./3sDS/train"
testPath = "./3sDS/test"
outPath = "./3sDS/test/out"
positivePath = "./3sDS/train/cough/Negative"
negativePath = "./3sDS/train/cough/Positive"


positiveNum = 0
negativeNum = 0
femaleNum = 0
maleNum = 0
shortNum = 0
midNum = 0
longNum = 0

# dir = "./data/train/Negative/388_Negative_female_22_cough.wav"
# time1 = librosa.get_duration(filename=dir)
# print(time1)

def classifier_duration():
    for eachfile in os.listdir(testPath):
        # print(eachfile)
        # print(str(os.path.isdir(path + eachfile)))
        # print(eachfile.split('_')[1])
        # if(eachfile.split('_')[1] == "Negative"):
        #     negativeNum+=1
        # # print(eachfile.split('_')[2])
        # if(eachfile.split('_')[2] == "female"):
        #     femaleNum+=1
        # else:
        #     if(eachfile.split('_')[2] == "male"):
        #         maleNum+=1

        # thispath = str(path + "/" + eachfile.split('_')[1] + "/" + eachfile)
        thispath = testPath +'/' + eachfile
        # y, sr = librosa.load(thispath)
        # print(thispath)
        time = librosa.get_duration(filename=thispath)
        print(time)

        #make path
        short_path = testPath + '/' + 'short'
        mid_path = testPath + '/' + 'mid'
        long_path = testPath + '/' + 'long'
        if(time <= 5):
            if not os.path.exists(short_path):
                os.mkdir(short_path)
            shutil.copy(thispath, short_path)
            print(eachfile + ' successfully moved')
            shortNum+=1

        else:
            if(time > 10):
                if not os.path.exists(long_path):
                    os.mkdir(long_path)
                shutil.copy(thispath, long_path)
                print(eachfile + ' successfully moved')
                longNum+=1
            else:
                if not os.path.exists(mid_path):
                    os.mkdir(mid_path)
                shutil.copy(thispath, mid_path)
                print(eachfile + ' successfully moved')
                midNum += 1


# # for eachfile in os.listdir(negativePath):
# #     print(eachfile)
# #     # print(eachfile.split('_')[1])
# #     if(eachfile.split('_')[1] == "Positive"):
# #         positiveNum+=1
# #     # print(eachfile.split('_')[2])
# #     if(eachfile.split('_')[2] == "female"):
# #         femaleNum+=1
# #     else:
# #         if(eachfile.split('_')[2] == "male"):
# #             maleNum+=1
# #
# #     thispath = str(path + "/" + eachfile.split('_')[1] + "/" + eachfile)
# #     # y, sr = librosa.load(thispath)
# #     time = librosa.get_duration(filename=thispath)
# #     # print(time)
# #
# #     if(time <= 5):
# #         shortNum+=1
# #     else:
# #         if(time > 10):
# #             longNum+=1
# #         else:midNum+=1
# #
# # # fileNum = 0
# # # moveNum = 0
# # # for eachfile in os.listdir(path):
# # #     if(os.path.isfile(eachfile)):
# # #         fileNum+=1
# # #     moveNum = int(fileNum * 0.2)
# # #     print((moveNum))
# # # for eachfile in os.listdir(path):
# #
# #
# # print(positiveNum)
# # print(negativeNum)
# # print(femaleNum)
# # print(maleNum)
print(shortNum)
print(midNum)
print(longNum)


def cutwave(file_path,out_path):
    fileTypes = ["long","mid","short"]
    for fileType in fileTypes:
        type_path = file_path + '/' + fileType
        # print(type_path)
        files = os.listdir(type_path)
        for wave in files:
            wave_path = type_path +'/'+ wave
            # print(wave_path)
            audio = AudioSegment.from_file(wave_path, "wav")
            size = 3000  # 切割的毫秒数 10s=10000
            chunks = make_chunks(audio, size)  # 将文件切割为10s一块
            name = wave.split('.')[0]  # 去除后缀名，便于保存
            if (fileType == "long"):
                for i, chunk in enumerate(chunks):
                    if(i != 1): continue
                    print("long i:",i)
                    chunk_name = out_path + '/'+ name + "_{type}.wav".format(type=fileType)
                    print(chunk_name)
                    chunk.export(chunk_name, format="wav")
            else:
                for i, chunk in enumerate(chunks):
                    if(i != 0): continue
                    print("mid and short i:",i)
                    chunk_name = out_path +'/'+ name + "_{type}.wav".format(type=fileType)
                    print(chunk_name)
                    chunk.export(chunk_name, format="wav")

if __name__ == '__main__':
    cutwave(testPath,outPath)

