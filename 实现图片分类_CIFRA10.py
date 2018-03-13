# -*- coding: utf-8 -*-
# @Time : 18-3-12 上午10:29
# @Author : chen
# @Site : 
# @File : 实现图片分类_CIFRA10.py
# @Software: PyCharm

"""
本实验是进行较为复杂的ＣＮＮ运算
在数据集为cifra10的数据集
可能分类为：ariplane,automoile,bird,cat,deer,dog,frog,horse,ship,truck
需要说明的是一般而言数据集都会比较大，所以在tensorflow中是建立一个图像管道从文件中一次批量读取
"""

import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six import moves
from tensorflow.python.framework import ops

ops.reset_default_graph()

# 1. 定位当前的工作路径
abspath = os.path.abspath(__file__) # abspath返回绝对路径
dname = os.path.dirname(abspath)
os.chdir(dname) # 切换到当前路径

# Start a graph session
sess = tf.Session()

# 2. 设置模型参数
batch_size = 128 # 批量处理的数量
data_dir = 'temp' # 数据集存储的路径
output_every = 50
generations = 1000
eval_every = 500
image_height = 32 # 图片的大小
image_width = 32
crop_height = 24
crop_width = 24
num_channels = 3 # 颜色的通道数
num_targets = 10 # 目标分类的数量
extract_folder = 'cifar-10-batches-bin'

# Exponential Learning Rate Decay Params
# 这些参数与训练过程有关
learning_rate = 0.1
lr_decay = 0.1
num_gens_to_wait = 250.

# Extract model parameters
image_vec_length = image_height * image_width * num_channels
record_length = 1 + image_vec_length  # 这里加上1是因为还有一位label

# 3. 下载数据集，存入temp路径中
data_dir = 'temp'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# 这是个小技巧，避免重复下载数据集
data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
if os.path.isfile(data_file):
    pass
else:
    # 开始下载数据集
    def progress(block_num, block_size, total_size):
        progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
        print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")

    # urlretrieve()方法直接将远程数据下载到本地, 但不知道为什么在这里不显示出来
    # urllib.request.urlretrieve(url,filename=None,reporthook=None, data=None)
    # reprothook为一个回调函数，调用程序进行处理
    filepath, _ = moves.urllib.request.urlretrieve(cifar10_url, data_file, progress)
    # tarfile主要是用来处理压缩文件的
    tarfile.open(filepath, 'r:gz').extractall(data_dir)


#  定义一个cifar的reader这样就避免了数据集过大不能进行读取的困难
def read_cifar_files(filename_queue, distort_images=True):
    # FixedLengthRecordReader读取固定长度字节的记录, decode_raw操作可以讲一个字符串转换为一个uint8的张量
    # 这两个操作是一起进行的
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string, tf.uint8)
    # tf.slice函数进行input的部分抽取
    """
    从这段代码中我们可以得知:
    我们从数据集中抽取图像分类标签与图像，image_vec_length为图像的大小
    标签为数据集的第一个标记的结果
    """
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
    # 进行shape的修改
    image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]),
                                 [num_channels, image_height, image_width])

    # 注意下tf.transpose的用法特别是第二个参数的含义 url = "http://blog.csdn.net/uestc_c2_403/article/details/73350498"
    # image_extacted.shape()=[3，32，32], image_uint8image.shape()=[32, 32, 3]
    image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
    reshaped_image = tf.cast(image_uint8image, tf.float32)
    # tf.image.resize_image_with_crop_or_pad是进行图像裁剪的方式
    # final_image为最后放入程序中的图片
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)

    if distort_images:
        # 这里好像是进行图片的反转什么之类的
        # Randomly flip the image horizontally, change the brightness and contrast
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image, max_delta=63)
        final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)

    # Normalize whitening
    final_image = tf.image.per_image_standardization(final_image)
    return (final_image, image_label)


# 5. 这段代码是tensorflow中比较特殊的数据管道，通过这个管道我们可以乱序读取数据集
# 详细介绍：http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/reading_data.html
"""
创建的顺序：
1.文件名列表
2.可配置的 文件名乱序(shuffling)
3.可配置的 最大训练迭代数(epoch limit)
4.文件名队列
5.针对输入文件格式的阅读器
6.纪录解析器
7.可配置的预处理器
8.样本队列
"""
def input_pipeline(batch_size, train_logical=True):
    if train_logical:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1, 6)]
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    """
    tf.train.string_input_producer
    把我们需要的全部文件打包为一个tf内部的queue类型，
    之后tf开文件就从这个queue中取目录了
    """
    filename_queue = tf.train.string_input_producer(files) # 将获取的文件列表转换成张量队列
    image, label = read_cifar_files(filename_queue)

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    """
    min_after_dequeue 定义了随机取样的缓冲区的大小
    """
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)

    return (example_batch, label_batch)


# 6. 构建卷积神经网的模型，代码的核心部分
def cifar_cnn_model(input_images, batch_size, train_logical=True):
    def truncated_normal_var(name, shape, dtype): # 自己去定义一个生成截取正态分布的函数
        return (
        tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))

    def zero_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

    # First Convolutional Layer
    with tf.variable_scope('conv1') as scope:
        # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
        conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
        # We convolve across the image with a stride size of 1
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
        # 需要记住一点东西
        conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        # ReLU element wise
        relu_conv1 = tf.nn.relu(conv1_add_bias)

    # Max Pooling
    pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer1')

    # Local Response Normalization (parameters from paper)
    # paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
    norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')

    # Second Convolutional Layer
    with tf.variable_scope('conv2') as scope:
        # Conv kernel is 5x5, across all prior 64 features and we create 64 more features
        conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
        # Convolve filter across prior output with stride size of 1
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize and add the bias
        conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        # ReLU element wise
        relu_conv2 = tf.nn.relu(conv2_add_bias)

    # Max Pooling
    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')

    # Local Response Normalization (parameters from paper)
    norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')

    # Reshape output into a single matrix for multiplication for the fully connected layers
    reshaped_output = tf.reshape(norm2, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    # First Fully Connected Layer
    with tf.variable_scope('full1') as scope:
        # Fully connected layer will have 384 outputs.
        full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))

    # Second Fully Connected Layer
    with tf.variable_scope('full2') as scope:
        # Second fully connected layer has 192 outputs.
        full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
        full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))

    # Final Fully Connected Layer -> 10 categories for output (num_targets)
    with tf.variable_scope('full3') as scope:
        # Final fully connected layer has 10 (num_targets) outputs.
        full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
        full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)

    return (final_output)


# 7. 定义损失函数
def cifar_loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32)) # squeeze的主要作用还是进行降维处理
    # Calculate cross entropy from logits and targets
    """
    这里要重新理解下sparse_softmax_cross_entropy_with_logits()这个函数
    logits:预测结果，必须是一个（batch_size, targets_num)的张量，同时数据类型为float
    labels:真实结果，必须是一个（batch_size, )的张量，同时数据类型为int
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return (cross_entropy_mean)


# Train step
def train_step(loss_value, generation_num):
    # Our learning rate is an exponential decay after we wait a fair number of generations
    # 这里使用了指数衰减法，没看懂，好像是用来求最佳的学习率的
    model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
                                                     num_gens_to_wait, lr_decay, staircase=True)
    # Create optimizer
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return (train_step)


# 8. 典型的定义结果准确度的函数
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return (accuracy)


# 4. 正式开始，前面的运行时进行数据集的获取
print('Getting/Transforming Data.')
"""
以下是进行input_pipeline的操作，这是可以进行通用的
"""
# 初始化input_pipeline
images, targets = input_pipeline(batch_size, train_logical=True)
# Get batch test images and targets from pipline
test_images, test_targets = input_pipeline(batch_size, train_logical=False)

# Declare Model
print('Creating the CIFAR10 Model.')
# tf.Variable_scope()为共享变量，这样就不用重复声明变量
with tf.variable_scope('model_definition') as scope:
    # Declare the training network model
    model_output = cifar_cnn_model(images, batch_size)
    # This is very important!!!  We must set the scope to REUSE the variables,
    #  otherwise, when we set the test network model, it will create new random
    #  variables.  Otherwise we get random evaluations on the test batches.
    scope.reuse_variables() # 检测变量是否共享了
    test_output = cifar_cnn_model(test_images, batch_size)

# Declare loss function
print('Declare Loss Function.')
loss = cifar_loss(model_output, targets)

# Create accuracy function
accuracy = accuracy_of_batch(test_output, test_targets)

# Create training operations
print('Creating the Training Operation.')
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# Initialize Variables
print('Initializing the Variables.')
init = tf.global_variables_initializer()
sess.run(init)

# Initialize queue (This queue will feed into the model, so no placeholders necessary)
tf.train.start_queue_runners(sess=sess)

# Train CIFAR Model
print('Starting Training')
train_loss = []
test_accuracy = []
for i in range(generations):
    _, loss_value = sess.run([train_op, loss]) # 还有这种操作,_的用法就是这个返回值没有用

    writer = tf.summary.FileWriter('./cnn_cifra10', sess.graph)

    if (i + 1) % output_every == 0:
        train_loss.append(loss_value)
        output = 'Generation {}: Loss = {:.5f}'.format((i + 1), loss_value)
        print(output)

    if (i + 1) % eval_every == 0:
        [temp_accuracy] = sess.run([accuracy])
        test_accuracy.append(temp_accuracy)
        acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100. * temp_accuracy)
        print(acc_output)
writer.close()

# Print loss and accuracy
# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)

# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()