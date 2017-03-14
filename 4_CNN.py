# coding:utf-8
# CNN 卷积神经网络

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


# 卷积函数
def conv2d(img, w, b):
    # relu 启动函数，bias_add 加上偏移量
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


# 池化函数
def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1],strides=[1, k, k, 1],padding='SAME')

# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 相关参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 输入尺寸 28*28 = 784
n_input = 784
# Label的数量
n_classes = 10

# 丢弃率
dropout = 0.75

# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
# reshape成需要的张量(输入的图像：长度为784的一维数组，扩展维度为28*28)，最终成为一个4维张量：图片数量*高*宽*颜色数
_x = tf.reshape(x, shape=[-1, 28, 28, 1])
# 输出张量，即每张图片对应的Label
y = tf.placeholder(tf.float32, [None, n_classes])

# 第一层卷积
# 第一层共享的卷积权重，即卷积核, 5*5大小*1颜色，*32个卷积数量，即特征数
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
# 第一层共享的偏差
bc1 = tf.Variable(tf.random_normal([32]))
# 第一层卷积
conv1 = conv2d(_x, wc1, bc1)
# 池化第一层，使用2*2
conv1 = max_pool(conv1, k=2)

# 丢弃操作, 减少过度匹配
keep_prob = tf.placeholder(tf.float32)
conv1 = tf.nn.dropout(conv1, keep_prob)

# 第二层卷积
# 第二层卷积共享权重，5*5大小，*32个特征图谱(来自第一层输出)*64个本层特征图谱
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
# 第二层共享偏差
bc2 = tf.Variable(tf.random_normal([64]))
# 第二层卷积
conv2 = conv2d(conv1, wc2, bc2)
# 第二层池化
conv2 = max_pool(conv2, k=2)
# 丢弃操作, 减少过度匹配
conv2 = tf.nn.dropout(conv2, keep_prob)

# 密集连接层，用来处理整个图像
# 权重与偏差
# 7*7*64 第二层卷积池化之后最终输出：7*7*64个特征图谱，含有1024个神经元
wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
bd1 = tf.Variable(tf.random_normal([1024]))

# 将第二层reshape成向量,wd1.get_shape():(3136,1024),as_list:[3136, 1024],[0]:3136
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
# 构建模型：乘上权重，加上偏差，并使用relu作为启动函数
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
# 减少过度匹配
dense1 = tf.nn.dropout(dense1, keep_prob)

# 输出层
# 定义输出层权重与偏差
# 1024(密集连接层的神经元数量)*10(图片的label类型数量)
wout = tf.Variable(tf.random_normal([1024, n_classes]))
bout = tf.Variable(tf.random_normal([n_classes]))

# 输出模型：计算凭证，每张图片属于某个label， *权重+偏差
pred = tf.add(tf.matmul(dense1, wout), bout)
# 成本函数，使用softmax函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 最小化成本函数
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估用张量
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 训练
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # 学习精度
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # 损耗
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})
            print('Iter %d, Minibatch Loss = %.6f , Training Accuracy = %.5f' % (step*batch_size, loss, acc))
        step += 1

    print('Optimization Finished!')
    # 通过测试集的前256个图像测试
    test_xs, test_ys = mnist.test.next_batch(256)
    print('Testing Accuracy: %.5f' % sess.run(accuracy
                   , feed_dict={x: test_xs, y: test_ys, keep_prob: 1.})
          )

    # 预测结果
    print(sess.run(tf.argmax(pred, 1)
                   , feed_dict={x: test_xs, keep_prob: 1.}))
    # 对照值
    print(sess.run(tf.argmax(test_ys, 1)))