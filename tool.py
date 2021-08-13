#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zyq
# datetime:2021/8/1 下午3:52
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(tf.test.is_gpu_available())