# coding:utf-8
# 分型 - Julia
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义复平面 水平(实部)范围： -2~2 步长 0.005，垂直(虚部)范围： -2~2 步长 0.005
Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]
# 复数Z代表着这个平面上每一个点(像素)
Z = X+1j*Y
# 复平面定义为tf的一个常数
Z = tf.constant(Z.astype(np.complex64))
# 与该复数同纬度的变量1
zs = tf.Variable(Z)
# 与该复数同纬度的变量2 - 初始值为0
ns = tf.Variable(tf.zeros_like(Z, dtype='float32'))

# 初始化交互式Session
sess = tf.InteractiveSession()
# 初始化所有变量，并运行
tf.global_variables_initializer().run()

# Julia 集合 迭代公式： Z(n+1) = Z(n)^2 - c
# 定义c初始值
c = complex(0.0, 0.75)
# 迭代函数
zs_ = tf.multiply(zs, zs) - c  # zs * zs - c
# 迭代停止条件：绝对值小于4
not_diverged = tf.abs(zs_) < 4
# 通过group组合多个运算
# 1.执行一步(step)迭代: Z(n+1)=Z(n)^2-c 并计算值
# 2.将这个值加到ns相关元素的变量中
step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))

for i in range(200) : step.run()

plt.imshow(ns.eval())
plt.show()
