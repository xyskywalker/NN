# coding:utf-8
# 逻辑斯回归
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 训练阶段
training_epochs = 25
# 参数
learning_rate = 0.01
batch_size = 100
display_step = 1

# 输入张量 x : 存储MNIST中的图片，每张图片尺寸是 28*28=784
x = tf.placeholder(tf.float32, [None, 784])
# 对应的label张量 y (one hot)
y = tf.placeholder(tf.float32, [None, 10])
# 输入权重张量
W = tf.Variable(tf.zeros([784, 10]))

# 偏差量
b = tf.Variable(tf.zeros([10]))
# 计算一张给定的图片属于哪个类别的凭据，简单的用输入张量 x 乘以权重张量 W 再加上一个偏差量
evidence = tf.matmul(x, W) + b
# 使用softmax算法生成属于每个类别的概率的输出向量
activation = tf.nn.softmax(evidence)

# 计算交叉熵误差
cross_entropy = y * tf.log(activation)
# 成本函数
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))
# 使用梯度下降法对成本函数最小化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 用于可视化训练过程的对象
avg_set = []
epoch_set = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 每一代(epochs)为一个训练周期
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        if epoch % display_step == 0:
            print('Epoch: %04d cost = %.9f' % (epoch+1, avg_cost))
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)

    # 测试
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Model accuracy: %.9f' % accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    plt.plot(epoch_set,avg_set, 'o', label='Logistic Regression Training phase')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()