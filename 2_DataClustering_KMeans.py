# coding:utf-8
# 多分类 - KMeans算法
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def display_partition(x_values,y_values,assignment_values):
    labels = []
    colors = ["red","blue","green","yellow"]
    for i in range(len(assignment_values)):
      labels.append(colors[(assignment_values[i])])
    color = labels
    df = pd.DataFrame\
            (dict(x =x_values,y = y_values ,color = labels ))
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], c=df['color'])
    plt.show()

# 集群的点数
num_vectors = 1000
# 分区数
num_clusters = 4
# K-Means算法需要运算的步数
num_steps = 100

# 初始化数据结构
x_values = []
y_values = []

# 创建训练数据
for i in range(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4, 0.7))
        y_values.append(np.random.normal(0.2, 0.8))
    else:
        x_values.append(np.random.normal(0.6, 0.4))
        y_values.append(np.random.normal(0.8, 0.5))

# 转换成完整的序列(使用Python内置zip函数)
vector_values = list(zip(x_values, y_values))
# 转换为tf的常数
vectors = tf.constant(vector_values)

# plt.plot(x_values,y_values,'o',label='Input Data')
# plt.legend()
# plt.show()

# 生成4个初始中心点
# tf.shape() : 提取维度
n_samples = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0, n_samples))
# 确定四个随机索引
begin = [0,]
size = [num_clusters,]
size[0] = num_clusters
# 初始的中心点索引
centroid_indices = tf.slice(random_indices, begin=begin, size=size)
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))

# 成本函数使用欧几里得距离
# 扩展维度
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)
# 计算距离准备
vectors_subtraction = tf.subtract(expanded_vectors, expanded_centroids)
# 成本函数：计算欧几里得距离，这里没有做开方处理，但是效果是一样的，reduce_sum:在指定维度内求和，square:平方
euclidean_distances = tf.reduce_sum(tf.square(vectors_subtraction), 2)
# argmin : 取最小值所在索引
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

# dynamic_partition将各数据分配到最近的各群中
partitions = tf.dynamic_partition(vectors, assignments, num_clusters)
# 重新计算中心点 reduce_mean:tensor中各个元素的平均值，用它作为新的中心点，concat:连接多个张量
update_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0),0) for partition in partitions], 0)

# 初始化变量
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

for step in range(num_steps):
   _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

display_partition(x_values,y_values, assignment_values)