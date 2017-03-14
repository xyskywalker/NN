# coding:utf-8
# 读取MNIST数据集
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取MNIST数据集
mnist_images = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 读取前10张图和对应的值
pixels, real_values = mnist_images.train.next_batch(10)

print(real_values)

example_to_visualize = 5
image = pixels[example_to_visualize,:]
print(image.shape)
image = np.reshape(image,[28,28])
print(image.shape)

plt.imshow(image)
plt.show()