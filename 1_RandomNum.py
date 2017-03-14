# coding:utf-8
# 随机数
import tensorflow as tf
import matplotlib.pyplot as plt

# 均匀分布
'''
uniform = tf.random_uniform([100],minval=0,maxval=1,dtype=tf.float32)

with tf.Session() as session:
    print uniform.eval()
    plt.hist(uniform.eval(),normed=True)
    plt.show()
'''

# 正态分布
'''
uniform = tf.random_normal([100], mean=0, stddev=2)

with tf.Session() as session:
    print uniform.eval()
    plt.hist(uniform.eval(),normed=True)
    plt.show()
'''

# 种子产生随机数(相同种子在同一个步骤中总产生相同的随机数)
'''
uniform_with_seed = tf.random_uniform([1],seed=1)
uniform_without_seed = tf.random_uniform([1])

print '1st_Run'
with tf.Session() as NO1st_Session:
    print NO1st_Session.run(uniform_with_seed)
    print NO1st_Session.run(uniform_with_seed)
    print NO1st_Session.run(uniform_without_seed)
    print NO1st_Session.run(uniform_without_seed)

print '2nd_Run'
with tf.Session() as NO2nd_Session:
    print NO2nd_Session.run(uniform_with_seed)
    print NO2nd_Session.run(uniform_with_seed)
    print NO2nd_Session.run(uniform_without_seed)
    print NO2nd_Session.run(uniform_without_seed)
'''

# 蒙特卡罗方法-计算π

trials = 100
hits = 0

# 产生随机数，并且值位于正方形 [1-,1]*[-1,1] 内
x = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)
y = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)

pi = []

sess = tf.Session()

# 令圆半径为1，园面积即为pi，随机数所处于的正方形面积为4，遍历100*100次，计算有多少点落在了圆内部，这些点的数量就是圆面积，即是pi
with sess.as_default():
    print(x.eval())
    print(y.eval())

    print(x.eval())
    print(y.eval())

    for i in range(1,trials):
        for j in range(1,trials):
            if x.eval() ** 2 + y.eval() ** 2 < 1 :
                hits = hits + 1
                # 将点数折算成面积
                pi.append((4*float(hits)/i)/trials)

print(pi)
print(hits)
plt.plot(pi)
plt.show()

