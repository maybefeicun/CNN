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
# cifra_link = 'http://www.cs.toronto.edu/~kriz/cifra-10-python.tar.gz'
cifar_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
data_dir = 'temp'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. 下载相应的数据集
target_file = os.path.join(data_dir, 'cifra-10-python.tar.gz')
if not os.path.isfile(target_file):
    print('CIFAR-10 file not found. Downloading CIFAR data (Size = 163MB)')
    print('This may take a few minutes, please wait.')
    # 利用urlretrieve函数将数据集导入目标路径
    filename, headers = urllib.request.urlretrieve(cifar_link, target_file)

# 解压
tar = tarfile.open(target_file) # 打开指定路径
tar.extractall(path=data_dir) # 将文件解压至该路径下
tar.close()

# 3. 创建训练所需的文件夹结构
# 临时目录下有两个文件夹train_dir,validation_dir
# 每个文件夹下有10个子文件夹，分别存储10个目标文件
train_folder = 'train_dir'
if not os.path.isdir(os.path.join(data_dir, train_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, train_folder, objects[i])
        os.makedirs(folder)
test_folder = 'validation_dir'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, test_folder, objects[i])
        os.makedirs(folder)

# 4.
