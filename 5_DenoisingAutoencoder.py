# coding:utf-8

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


# 生成权重初始值
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    # 返回一个均匀分布
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


# 添加高斯噪音的自编码器
class AdditiveGaussianNoiseAutoencoder(object):
    # n_input:输入变量
    # n_hidden:隐含层节点数
    # transfer_function隐含层激活函数，默认tf.nn.softplus
    # optimizer:优化器，默认tf.train.AdamOptimizer()
    # scale:高斯噪声系数，默认0.1
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus
                 , optimizer=tf.train.AdamOptimizer(), scale=0.1):
        # 相关参数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weight = self._initialize_weights()
        self.weights = network_weight

        # 网络定义
        # x:输入
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 定义隐含层
        # 给输入加上噪声：self.x+scale*tf.random_normal((n_input,))
        # 网络定义：加上噪声的输入x与权重w1相乘之后加上偏置b1，最后使用transfer调用激活函数
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),self.weights['w1']
                                                     )
                                           , self.weights['b1']
                                           )
                                    )
        # 定义输出层，重建操作
        # 隐含层的输出*权重w2+偏置b2，输出层不用激活函数
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        # 损失函数，使用平方误差
        # subtract 计算输出与输入的差
        # pow 平方
        # reduce_sum 求和
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x),2.0))
        # 优化函数
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 参数初始化函数
    def _initialize_weights(self):
        all_weight = dict()
        all_weight['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weight['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weight['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weight['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weight

    # 训练函数
    def partial_fit(self, X):
        # 运行两个tensor，cost:损失函数，optimizer:优化函数
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 单独的运行损失函数
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 返回隐含层的结果
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 返回输出层，即将隐含层的输出作为输入，重整之后返回
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 获取隐含层的权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐含层的偏置
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# 对测试数据进行标准化处理的函数，转换成0均值，标准差为1。
# 处理方法就是先减去均值再除以标准差。直接使用sklearn.preprocessing的StandardScaler类
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# 随机获取block数据，在0~(len(date)-batch_size)之间取一个随机整数座位block的起点，然后取batch_size尺寸的数据
# 这是不放回抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]


# 标准化训练数据
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 设定参数
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1


# 实例化自编码器
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)


for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size

    if epoch%display_step == 0:
        print("Epoch: %04d cost = %.9f" % ((epoch+1), avg_cost))

print('Total cost: %.9f' % autoencoder.calc_total_cost(X_test))

'''
import matplotlib.pyplot as plt
batch_xs = get_random_block_from_data(X_train, batch_size)
image = batch_xs[1,:]
image = np.reshape(image,[28,28])
plt.imshow(image)
plt.show()
print(image)

hidden = autoencoder.transform(batch_xs)
generate = autoencoder.generate(hidden)
image = generate[1,:]
image = np.reshape(image,[28,28])
plt.imshow(image)
plt.show()
print(image)
'''