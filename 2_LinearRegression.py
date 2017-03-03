# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 线性回归

# 准备测试数据
number_of_points = 500

x_point = []
y_point = []

# y=ax+b，定义常数a,b
a = 0.22
b = 0.78

for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5) #uniform(0,0.5) #.normal(0.0, 0.5)
    y = a*x + b + np.random.normal(0.0, 0.1) #uniform(0,0.1)#.normal(0.0, 0.1)
    x_point.append([x])
    y_point.append([y])

# plt.plot(x_point, y_point, 'o', label='Input Data')
# plt.legend()
# plt.show()

# 通过梯度下降法 求 线性方程 y=ax+b 中的 a,b

# 定义未知数 a,b ，a定义为介于1 ~ -1之间的随机数, b初始值定义为0
a_ = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b_ = tf.Variable(tf.zeros([1]))

# 定义线性方程 y=axx+b
y = a_ * x_point + b_

# 定义成本函数，本例中的意义就是表明了a与b的正确程度，这里使用了均方误差(Mean Squared Error, MSE)
cost_function = tf.reduce_mean(tf.square(y - y_point))

# 通过梯度下降法来使得成本函数值最小化
# 0.5 为学习率
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost_function)

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    # 迭代21次，每次执行一遍优化步骤
    for step in range(0,21):
        session.run(train)
        # 每五部打印一个结果
        if(step%5) == 0:
            plt.plot(x_point,y_point,'o',label='step = {}'.format(step))
            plt.plot(x_point,session.run(a_) * x_point + session.run(b_))
            plt.legend()
            plt.show()

    print 'a = {}'.format(session.run(a_))
    print 'b = {}'.format(session.run(b_))
