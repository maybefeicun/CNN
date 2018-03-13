# -*- coding: utf-8 -*-
# @Time : 18-3-7 下午4:10
# @Author : chen
# @Site : 
# @File : 实现简单的CNN_MNIST.py
# @Software: PyCharm

"""
本代码是利用ＭＮＩＳＴ数据集来实现一个四层的卷积神经网络
前两个卷积层由Convolution-Relu-maxpool操作组成
后面两层则是全联接层
"""
from CNN.mnist_loader import load_data_wrapper
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()
summary_write = tf.summary.FileWriter('tensorboard', tf.get_default_graph())

# 1. 获取数据集
"""
这里的
train_xdata.shape()=28*28
train_labels.shape()=1*10
数据集的大小不确定

这里要说明下数据的结构
train_xdata：为图片的数据，shape=[50000, 28, 28]
trian_labels:为图片所代表的数字，shape=[50000, 10]
"""
train_xdata, train_labels, test_xdata, test_labels = load_data_wrapper()
train_xdata = np.array(train_xdata)
train_labels = np.array(train_labels)
test_xdata = np.array(test_xdata)
test_labels = np.array(test_labels)

# 2. 设置模型参数，图像已经灰度图
batch_size = 100 # 训练时的批量大小
learning_rate = 0.005
evaluation_size = 500 # 测试时抽取数据集的大小
image_width = train_xdata[0].shape[0] # 图像大小
image_height = train_xdata[0].shape[1] #
target_size = max(train_labels) + 1 # 这里要理解下,不能使用argmax()
num_channels = 1 # 图像为单通道的
generations = 300 # 迭代次数
eval_every = 5 # 循环判断所用,没什么用
conv1_features = 25 # 第一层卷积的feature map的数量
conv2_features = 20 # 第二层卷积的feature map的数量
max_pool_size1 = 2 # 池化层的大小
max_pool_size2 = 2 # 池化层的大小
fully_connected_size1 = 100 # 全联接层的大小

# 3. 声明变量以及占位符
"""
x_input:训练集
eval_input:测试集
"""
x_input_shape = (batch_size, image_width, image_height, num_channels) # 卷积层的输入格式
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size)) # 这个要注意
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape=(evaluation_size))

# 4. 声明卷积层的权重和偏置的参数
"""
本实验设置了两个卷积层, 分别初始化两个层的weights与bias
这里要注意weights张量的shape初始化的数据
tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差
"""
conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features],stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

"""
一定要注意summary函数必须要有name与value两个变量
"""
with tf.name_scope('conv1_weight'):
    tf.summary.histogram(name="conv1_weight", values=conv1_weight)
with tf.name_scope('conv1_bias'):
    tf.summary.histogram(name="conv1_bias", values=conv1_bias)
# 5. 声明全联接层的权重和偏置
"""
本段代码我们首先要注意全联接层的结果大小的算法

"""
resulting_width = image_width // (max_pool_size1 * max_pool_size2) # 运算符//表示为Ｃ语言中的除法运算
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_width * resulting_height * conv2_features # 这里这么写是有原因的
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

# 6. 声明算法模型,定义训练模型（也可以说是训练过程）
"""
代码的核心
每次卷积结束后都会进行一个池化过程
"""
def my_conv_net(input_data):
    # First Conv-ReLU-MaxPool Layer
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                               strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')
    """
    input_data.shape() = [100, 28, 28, 1]
    conv1_weight.shape() = [4, 4, 1, 25]
    conv1.shape() = [100, 28, 28, 25] , 可以这么理解这里的conv1相当于一张图片但是通道改变了
    max_pool1.shape() = [100, 14, 14, 25]
    上面的这段代码是ＣＮＮ，卷积的一个常规方法，他的步骤是：
    １．先进行卷积运算
    ２．再加上偏置，并做一个relu处理（非线性化处理）
    ３．最后进行池化处理（达到降维的效果）
    """

    # Second Conv-ReLU-MaxPool Layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

    # Transform Output into a 1xN layer for next fully connected layer
    """
    max_pool.get_shape().as_list()是将张量的形状以列表的形式输出来,这样就成了一个一维的向量
    这麽做的原因是想修改最后一个卷积层的shape,使得卷积层的结果能和全联接层进行运算
    """
    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3] # 这里不理解
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    # First Fully Connected Layer
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

    # Second Fully Connected Layer
    """
    到这里整个图片就已经被１维化了
    """
    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

    return (final_model_output)

# 7. 声明训练模型
model_output = my_conv_net(input_data=x_input)
test_model_output = my_conv_net(input_data=eval_input)

with tf.name_scope('model_output'):
    tf.summary.histogram('model_output', model_output)

# 8. 定义损失函数
"""
这个定义损失函数的方法应该是固定的，需要十分注意
没理解
"""
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(model_output, y_target))
# 上面的写法是错误的
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
with tf.name_scope('loss'):
    tf.summary.scalar('loss', loss)
# 9. 创建测试集与训练集的预测函数,同时创建对应的准确度函数，评估模型的准确度
prediction = tf.nn.softmax(model_output)
# with tf.name_scope('prediction'):
#     tf.summary.histogram(prediction)
test_prediction = tf.nn.softmax(test_model_output)
# with tf.name_scope('test_prediction'):
#     tf.summary.scalar('test_prediction')
"""
创建准确度函数
这里要注意在tensorflow中logits的含义一般为模型的逻辑输出
"""
def get_accuracy(logits, targets): # 这个函数就是用来求准确率的其他没什么
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return (100. * num_correct / batch_predictions.shape[0])

# 10. 创建优化器，声明训练步长，初始化所有模型变量
my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()
sess.run(init)

# 11. 开始训练模型
train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)
    summary = sess.run(summary_op, feed_dict=train_dict)
    write = tf.summary.FileWriter("./graphs_1", sess.graph)
    write.add_summary(summary, (i + 1))
    # output = sess.run(model_output, feed_dict=train_dict)
    # print(output)

    if (i + 1) % eval_every == 0:
        eval_index = np.random.choice(len(test_xdata), size=evaluation_size)
        eval_x = test_xdata[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]
        test_dict = {eval_input: eval_x, eval_target: eval_y}
        test_preds = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)
        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
        """
        这里使用了round函数，他表示以一定的小说点位数取数
        """
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
write.close()

# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()