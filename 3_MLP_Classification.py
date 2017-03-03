# coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 多层感知分类器

# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 定义参数
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# 第一层神经元数量
n_hidden_1 = 256
# 第二层神经元数量
n_hidden_2 = 256

# 输入尺寸 一张图片为 28*28 = 784
n_input = 784

# 输出的类别数量
n_classes = 10

# 输入层
# 输入的图像
x = tf.placeholder(tf.float32, [None, n_input])
# 对应的类别
y = tf.placeholder(tf.float32, [None, n_classes])

# 定义第一个隐藏层
# h 权重 尺寸：784(输入，图像尺寸) * 256(第一层的节点数)
h = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
# 对应的偏差张量
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))
# 定义第一层 使用sigmoid作为神经元的启动函数, 求和(x*h+bias_layer_1)
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, h), bias_layer_1))

# 定义第二个隐藏层
# w 权重 尺寸: 256(输入，即第一层的输出) * 256(第二层的节点数)
w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
# 偏差张量
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))
# 定义第二层
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w), bias_layer_2))

# 定义输出层
# 权重，尺寸 256(输入，即第二层的输出) * 10(类别数量)
output = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# 偏差量
bias_output = tf.Variable(tf.random_normal([n_classes]))
# 定义输出层 输入(第二个隐藏层)*权重 + 偏差量， matmul: 向量乘法
output_layer = tf.matmul(layer_2, output) + bias_output

# 成本函数 reduce mean 降维->平均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
# 使用了Adam算法来最小化成本函数
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 用于可视化训练过程的对象
avg_set = []
epoch_set = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        if epoch % display_step == 0:
            print 'Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost)
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print 'Training phase finished'

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print 'Model accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    # 显示收敛过程
    plt.plot(epoch_set,avg_set, 'o', label='MLP Training phase')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()




