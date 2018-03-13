# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 17:08
# @Author  : chen
# @File    : infection_cnn.py
# @Software: PyCharm

"""
这次的代码是利用了google已有的inception CNN模型
同时仍然使用cifra10数据集进行测试使用
"""

import os
import tarfile
import _pickle as cPickle
import numpy as np
import urllib.request
import scipy.misc

# 1. 下载所需的数据集
cifra_link = 'http://www.cs.toronto.edu/~kriz/cifra-10-python.tar.gz'
data_dir = 'temp'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. 下载相应的数据集