# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.image as mp_image
import matplotlib.pyplot as plt
################################################
# x=1
# y=x+9

# print(y)

# x=tf.constant(1,name='x')
# y=tf.Variable(x+9,name='y')

# print(y)
################################################
# x=tf.constant(1,name='x')
# y=tf.Variable(x+9,name='y')

# model = tf.initialize_all_variables()

# model = tf.global_variables_initializer()

# with tf.Session() as session:
#     session.run(model)
#     print(session.run(y))
################################################
# a = tf.placeholder('int32')
# b = tf.placeholder('int32')

# y = tf.multiply(a,b)

# sess = tf.Session()

# print sess.run(y,feed_dict = {a:2 , b:5})
################################################
# TensorBoard
#
# a = tf.constant(10, name='a')
# b = tf.constant(90, name='b')

# y = tf.Variable(a+b*2, name='y')
# model = tf.global_variables_initializer()

# with tf.Session() as session:
#     merged = tf.summary.merge_all()
#     writer = tf.summary.FileWriter('/tmp/tensorflowlogs', session.graph)
#     session.run(model)
#     print session.run(y)
################################################
# 一维数组->张量
# tensor_1d = np.array([1.3, 1, 0.4, 23.99])
# print tensor_1d
# print tensor_1d.ndim
# print tensor_1d.shape
# print tensor_1d.size
# print tensor_1d.dtype

# tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

# with tf.Session() as session:
#     print session.run(tf_tensor)
#     print session.run(tf_tensor[0])
################################################
# 二维数组->张量
# matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)],dtype='int32')
# matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)],dtype='int32')
# matrix3 = np.array([(2,7,2),(1,4,2),(9,0,2)],dtype='float32')

# print matrix1
# print matrix2

# matrix1 = tf.constant(matrix1)
# matrix2 = tf.constant(matrix2)

# matrix_product = tf.matmul(matrix1, matrix2)
# matrix_sum = tf.add(matrix1, matrix2)
# matrix_det = tf.matrix_determinant(matrix3)

# with tf.Session() as sess:
#     result1 = sess.run(matrix_product)
#     result2 = sess.run(matrix_sum)
#     result3 = sess.run(matrix_det)

# print result1
# print result2
# print result3

################################################
# 图像分割
# filename = 'packt.png'

# input_image = mp_image.imread(filename)

# print format(input_image.ndim)
# print format(input_image.shape)

# plt.imshow(input_image)
# plt.show()

# my_image = tf.placeholder('float32',[None,None,3])
# slice = tf.slice(my_image,[10,0,0],[16,-1,-1])

# with tf.Session() as session:
#     result = session.run(slice,feed_dict={my_image: input_image})
#     print format(result.ndim)
#     print format(result.shape)

# plt.imshow(result)
# plt.show()
################################################
################################################
# 图像转置
'''
filename = 'packt.png'

input_image = mp_image.imread(filename)

x = tf.Variable(input_image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1,0,2])
    session.run(model)
   result = session.run(x)

plt.imshow(result)
plt.show()
'''

################################################
# 复数
'''
x = 5.+4j
print x

x = complex(6,7)
print x

print x.real
print x.imag
'''

################################################
# 计算梯度
'''
x = tf.placeholder(tf.float32)

y = 2 * x * x

# tf内的梯度计算函数
var_grad = tf.gradients(y,x)

with tf.Session() as session:
    # 计算 x=1 处的函数梯度
    var_grad_val = session.run(var_grad, feed_dict={x:1})
    print var_grad_val
'''