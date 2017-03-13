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
# 输出层权重和偏置，输出层10个节点(即label数量)，初始化为0
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# 定义输入层
x = tf.placeholder(tf.float32, [None, in_units])
# 用于Dropout的keep_prob(也就是保留节点的概率)。训练时小于1，避免过拟合，预测时等于1，全部保留
keep_prob = tf.placeholder(tf.float32)

# 定义隐含层
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob=keep_prob)
# 定义输出层
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
# 对比值
y_ = tf.placeholder(tf.float32, [None, 10])
# 成本函数(交叉熵)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 优化器
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 变量初始化
tf.global_variables_initializer().run()
# 训练模型
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 测试模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 测试时保留率为1
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
