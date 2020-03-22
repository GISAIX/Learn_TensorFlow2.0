import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIABLE_DEVICES']='O'

import tensorflow as tf 
import numpy as np
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in phy_gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

a = tf.constant([[1,1],[1,1]])
b = tf.constant([[1,0],[0,2]])

# 矩阵乘法：
c = tf.matmul(a,b)
print(c.numpy())
print("\n")

'''
[[1 2]
 [1 2]]
'''

# 向量乘法：
d = a*b
print(d.numpy())
print("\n")

'''
[[1 0]
 [0 2]]
'''

# 均值
m = [1,2,3,4,5]
mean = tf.reduce_mean(m)
print(mean)
print("\n")

'''
tf.Tensor(3, shape=(), dtype=int32)
'''

# 方差
x = [1,3,5,7,9]
y = [1,1,1,1,1]
r = tf.math.squared_difference(x,y)
print(r)
print("\n")

'''
tf.Tensor([ 0  4 16 36 64], shape=(5,), dtype=int32)
'''

# 随机数生成

# 正态分布
rand1 = tf.random.normal(shape=(3,2),mean=0.0,stddev=2.0)
print(rand1)
print("\n")

'''
tf.Tensor(
[[-0.9260874  4.9333167]
 [-2.0966494  2.6556277]
 [ 1.8420827  1.7115388]], shape=(3, 2), dtype=float32)
'''

# 均匀分布
rand2 = tf.random.uniform(shape=(2,4),minval=1,maxval=10)
print(rand2)
print("\n")

'''
tf.Tensor(
[[4.1472673 1.3256727 5.5261173 1.2175164]
 [7.5022736 6.664027  7.76919   7.363538 ]], shape=(2, 4), dtype=float32)
'''

# 寻找极值

a = [1,3,5,2,11,0,99]
# 极大值
d = tf.argmax(a)
print(d)
# 极小值
e = tf.argmin(a)
print(e)