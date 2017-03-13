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

