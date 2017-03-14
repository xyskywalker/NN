# coding:utf-8
# 简单的卷积网络 - MNIST数据集分类

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()


# 权重初始化函数
def weight_variable(shape):
    # 添加一些随机噪声来避免完全对称，使用截断正态分布，标准差为0.1
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)


# 偏置量初始化函数
def bias_variable(shape):
    # 为偏置量增加一个很小的正值(0.1)，避免死亡节点
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)


# 卷积函数
def conv2d(x, W):
    # x: 输入
    # W: 卷积参数，例如[5,5,1,32]：5,5代表卷积核尺寸、1代表通道数：黑白图像为1，彩色图像为3、32代表卷积核数量也就是要提取的特征数量
    # strides: 步长，都是1代表会扫描所有的点
    # padding: SAME会加上padding让卷积的输入和输出保持一致尺寸
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 最大池化函数
def max_pool_2x2(x):
    # 使用2*2进行最大池化，即把2*2的像素块降为1*1，保留原像素块中灰度最高的一个像素，即提取最显著特征
    # 横竖两个方向上以2为步长
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 输入
x = tf.placeholder(tf.float32, [None, 784])
# 对应的label
y_ = tf.placeholder(tf.float32, [None, 10])
# 将一维的输入转成二维图像结棍
# -1: 数量不定
# 28*28: 图像尺寸
# 1: 通道数，黑白图像为1
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层
# 共享权重，尺寸：[5, 5, 1, 32]，5*5的卷积核尺寸、1个颜色通道、32个卷积核数量，即特征数量
W_conv1 = weight_variable([5, 5, 1, 32])
# 共享偏置量，32个
b_conv1 = bias_variable([32])
# 激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化
h_conv1 = max_pool_2x2(h_conv1)

