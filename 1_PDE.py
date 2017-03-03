# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.image as mp_image
import matplotlib.pyplot as plt

# 偏微分方程，池塘中的水滴

def make_kernel(a):
    a_ = np.asarray(a)
    a_ = a_.reshape(list(a_.shape) + [1,1])
    return tf.constant(a_, dtype=1)


def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1,1,1,1],padding='SAME')
    return y[0, :, :, 0]


def laplace(x):
    laplace_k = make_kernel([
        [0.5, 1.0, 0.5],
        [1.0, -6., 1.0],
        [0.5, 1.0, 0.5]
    ])
    return simple_conv(x, laplace_k)


# 500*500的池子
N = 500

# 初始条件，一个二维张量
u_init = np.zeros([N,N], dtype=np.float32)

# 随机创建40个雨滴
for n in range(40):
    a,b = np.random.randint(0, N, 2)
    u_init[a,b] = np.random.uniform()

# plt.imshow(u_init)
# plt.show()

# 定义一个新张量，表示随着时间t改变时，池子的状况
ut_init = np.zeros([N,N], dtype=np.float32)

# 时间同步
eps = tf.placeholder(tf.float32, shape=())
# 阻尼系数
damping = tf.placeholder(tf.float32, shape=())

# 定义两个tf中的变量
U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# PDE模型
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# 定义运算组->池子按时间t的变化
step = tf.group(U.assign(U_), Ut.assign(Ut_))

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(1000):
        session.run(step,feed_dict={eps: 0.03, damping: 0.04})
        if i % 50 == 0 :
            plt.imshow(U.eval())
            plt.show()