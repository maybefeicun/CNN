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
data_dir = 'temp_infection' # 创建一个存储数据集的主文件夹
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# 表明对象的范围
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. 下载相应的数据集
target_file = os.path.join(data_dir, 'cifra-10-python.tar.gz')
if not os.path.isfile(target_file):
    print('CIFAR-10 file not found. Downloading CIFAR data (Size = 163MB)')
    print('This may take a few minutes, please wait.')
    # 利用urlretrieve函数将数据集导入目标路径
    filename, headers = urllib.request.urlretrieve(cifar_link, target_file)

# 3. 解压
tar = tarfile.open(target_file) # 打开指定路径
tar.extractall(path=data_dir) # 将文件解压至该路径下
tar.close()

# 4. 创建训练所需的文件夹结构
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

# 4. 创建函数实现从内存加载图片存入文件夹中
def load_batch_from_file(file):
    file_conn = open(file, 'rb') # 说明文件是以二进制的方式进行存储的
    # load是从文件中加载图片
    image_diction = cPickle.load(file_conn, encoding='latin1') # cPickle是pickle的一个C语言版的更快速的方法
    file_conn.close()
    return(image_diction)

def save_image_from_dict(image_dict, folder='data_dir'):
    for ix, label in enumerate(image_dict['labels']):
        folder_path = os.path.join(data_dir, folder, objects[label])
        filename = image_dict['filenames'][ix]
        image_array = image_dict['data'][ix]
        image_array.resize([3, 32, 32])
        output_location = os.path.join(folder_path, filename)
        scipy.misc.imsave(output_location, image_array.transpose())

# 对于上一步的函数，遍历下载数据文件，并把每个图片保存到正确的位置
data_location = os.path.join(data_dir, 'cifar-10-batches-py')
train_names = ['data_batch_' + str(x) for x in range(1, 6)]
test_names = ['test_batch']
for file in train_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_image_from_dict(image_dict, folder=train_folder)
for file in test_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_image_from_dict(image_dict, folder=test_folder)

# 最后部分是创建标注文件，该文件用标注（而不是数值索引）自解释输出结果
cifar_labels_file = os.path.join(data_dir, 'cifar10_labels.txt')
print('Writing labels file, {}'.format(cifar_labels_file))
with open(cifar_labels_file, 'w') as labels_file:
    for item in objects:
        labels_file.write("{}\n".format(item))