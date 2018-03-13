# -*- coding: utf-8 -*-
# @Time : 18-3-8 上午11:05
# @Author : chen
# @Site : 
# @File : ceshi.py
# @Software: PyCharm

a = 4
b = 3
print(a / b)
print(a // b)

a = [1, 3, 4, 5]
print(max(a))

a = [[[1], [2]], [[3], [4]]]
print(max(a))

import tensorflow as tf

sess = tf.Session()

temp_var = tf.Variable(tf.constant(1., shape=[1, 2, 3], dtype=tf.float32))
print(temp_var.get_shape())

import numpy as np

x = [[1, 2], [2, 2]]
x = np.array(x)
x = x.reshape(4)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Load data
data_dir = 'temp'
mnist = read_data_sets(data_dir)

# Convert images into 28x28 (they are downloaded as 1x784)
train_xdata = np.array([np.reshape(x, (28,28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28,28)) for x in mnist.test.images])

# Convert labels into one-hot encoded vectors
train_labels = mnist.train.labels
test_labels = mnist.test.labels
pass
