# coding:utf-8

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取MNIST数据集
mnist_images = input_data.read_data_sets('MNIST_data/', one_hot=False)

# 取前100个作为训练数据
train_pixels, train_list_values = mnist_images.train.next_batch(100)

# 取前10个作为测试数据
test_pixels, test_list_values = mnist_images.test.next_batch(10)

# 定义张量-两个占位符，一个训练用，一个测试用
train_pixel_tensor = tf.placeholder('float', [None, 784])
test_pixel_tensor = tf.placeholder('float', [784])

# 成本函数：像素点的距离
distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.negative(test_pixel_tensor))), reduction_indices=1)
# 计算最小距离
pred = tf.arg_min(distance, 0)

# 准确度
accuracy = 0

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(len(test_list_values)):
        # 计算最近邻的距离
        nn_index = session.run(pred, feed_dict={train_pixel_tensor:train_pixels, test_pixel_tensor:test_pixels[i,:]})

        # predictedClass = np.argmax(train_list_values[nn_index])
        # trueClass = np.argmax(test_list_values[i])
        predictedClass = train_list_values[nn_index]
        trueClass = test_list_values[i]

        print 'Test N ', i , 'Predicted Class: ', predictedClass, 'True Class: ', trueClass
        if predictedClass == trueClass:
            accuracy += 1./len(test_pixels)

    print 'Result = ' , accuracy


