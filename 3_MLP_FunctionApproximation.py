# coding:utf-8

import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt

# 准备训练数据(cos函数)
NUM_points = 1000
np.random.seed(NUM_points)
function_to_learn = lambda x: np.cos(x) + 0.1*np.random.rand(*x.shape)

# 隐藏层 10个神经元
layer_1_neurons = 10

# 每次学习的批量尺寸
batch_size = 100
# 学习的代数
NUM_EPOCHS = 1500

# 创建训练集和测试集
# 创建1000个随机的x值，用来计算1000个y值
all_x = np.float32(np.random.uniform(-2*math.pi, 2*math.pi, (1, NUM_points))).T
np.random.shuffle(all_x)
train_size = int(900)

# 900个训练集
x_training = all_x[:train_size]
y_training = function_to_learn(x_training)

# 100个测试集
x_validation = all_x[train_size:]
y_validation = function_to_learn(x_validation)

# 显示这些集合
# plt.figure(1)
# plt.scatter(x_training, y_training, c='blue', label='train')
# plt.scatter(x_validation, y_validation, c='red', label='validation')
# plt.legend()
# plt.show()

# 建立模型
X = tf.placeholder(tf.float32, [None, 1], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

# 建立隐藏层 1(输入点的x值) * 10(隐藏层的神经元数量)
# 权重
w_h = tf.Variable(tf.random_uniform([1, layer_1_neurons], minval=-1, maxval=1, dtype=tf.float32))
# 偏移量
b_h = tf.Variable(tf.zeros([1, layer_1_neurons], dtype=tf.float32))
# 神经元启动函数sigmoid
h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)

# 输出层 10(隐藏层的神经元数量) * 1(输出的y值)
# 权重
w_o = tf.Variable(tf.random_uniform([layer_1_neurons, 1], minval=-1, maxval=1, dtype=tf.float32))
# 偏移量，输出 1*1
b_o = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))
# 定义输出层模型
model = tf.matmul(h, w_o) + b_o
# 成本函数 l2 loss 等效： ((model - Y)^2)/2
cost_fun = tf.nn.l2_loss(model - Y)
# 使用了Adam算法来最小化成本函数
train_op = tf.train.AdamOptimizer().minimize(cost_fun)

# 运行模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())

errors = []
for i in range(NUM_EPOCHS):
    for start, end in zip(range(0, len(x_training), batch_size), range(batch_size, len(x_training), batch_size)):
        sess.run(train_op, feed_dict={X: x_training[start:end], Y: y_training[start:end]})
    cost = sess.run(cost_fun, feed_dict={X: x_validation, Y: y_validation})
    # cost = sess.run(tf.nn.l2_loss(model - y_validation), feed_dict={X:x_validation})
    errors.append(cost)
    if i%100 == 0:
        print 'Epoch %d, cost = %g' % (i, cost)

y_forecast = sess.run(model, feed_dict={X: x_validation})

plt.figure(1)
plt.scatter(x_validation, y_forecast, c='red', label='forecast')
plt.scatter(x_validation, y_validation, c='blue', label='validation')
plt.legend()
plt.show()

# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, 'my-model')

# plt.plot(errors, label='MLP Function Approximation')
# plt.xlabel('epochs')
# plt.ylabel('cost')
# plt.legend()
# plt.show()