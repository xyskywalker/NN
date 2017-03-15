# coding:utf-8
# 卷积网络 - CIFAR10 数据集分类

import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 3000
batch_size = 128
data_dir = 'tmp/cifar-10-batches-bin'


# 权重初始化函数，给权重加上损失，避免过拟合
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# 获取训练数据，这里会对原始图像进行增强操作，包括随机的水平翻转，随机剪切24*24的大小，随机设置亮度和对比度
# 最后进行标准化处理(数据减去均值，除以方差)，保证0均值，方差为1
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
# 获取测试数据，不做特别的处理，仅裁剪正中的24*24区块，并进行标准化操作
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 定义数据输入的placeholder，因为每一批输入的图像数量(batch_size)在网络定义中需要使用，不能再设置一个None
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层卷积
# 权重初始化，卷积核5*5，3个通道，64个卷积核(64个特征)，标准差为0.05，不做L2正则，weight loss设置为0
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
# 卷积操作，
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
# 偏置初始化为0
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# 加上偏置之后应用激活函数
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
# 最大池化，池化尺寸3*3，步长2*2，最大池化和步长不一致，可以增加数据丰富性
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# LRN处理，模仿生物神经系统的"侧抑制"，增强模型泛化能力。响应比较大的值变得更大，抑制其他反馈比较小的神经元
# 适合于relu这种无上边界的函数，不适用于sigmoid这样本身带有固定边界能够抑制过大值的激活函数
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# 第二层卷积
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
# 第二层的偏置都初始化为 0.1
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
# 第二层卷积调换LRN和最大池化操作步骤
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一个全连接层
# 数据扁平化
reshape = tf.reshape(pool2, [batch_size, -1])
# 获取扁平化之后的长度
dim = reshape.get_shape()[1].value
# 权重，这一层带有384个神经元，正态分布标准差0.04，weight loss设置为0.004避免过拟合
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第二个全连接层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 输出层
# 权重，标准差是上一个全连接层神经元数量的倒数
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

# 损失计算
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    # 计算交叉熵(带权重)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits
                                                                   , labels=labels
                                                                   , name='cross_entropy_pre_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# 损失函数
loss = loss(logits, label_holder)
# 优化器
train_op = tf.train.AdagradOptimizer(1e-3).minimize(loss)

# 求准确率(top 1)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 启动线程，用于图片载入操作
tf.train.start_queue_runners(sess=sess)

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step%10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        print('Step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)'
              % (step, loss_value, examples_per_sec, sec_per_batch))

# 准确率测试
num_examples = 10000
num_iter = int(math.ceil(num_examples/batch_size))

true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

prediction = float(true_count) / float(total_sample_count)
print('Prediction @ 1 = %.6f' % prediction)





