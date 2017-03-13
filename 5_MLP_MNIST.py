# coding:utf-8
# 多层感知器-对MNIST进行分类

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()

# 输入数(图像的长度)
in_units = 784
# 隐含层节点数
h1_units = 300
# 隐含层权重和偏置
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
# 输出层权重和偏置，输出层10个节点，初始化为0
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

