import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIABLE_DEVICES']='O'

import tensorflow as tf 
import numpy as np
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in phy_gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

# tensor
a = tf.constant([[1,2],[3,4]])
b = a*a
print(a)
print(b)
print(type(a))
print(type(b))
print("\n")

'''
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32) 
tf.Tensor(
[[ 1  4]
 [ 9 16]], shape=(2, 2), dtype=int32) 
<class 'tensorflow.python.framework.ops.EagerTensor'> 
<class 'tensorflow.python.framework.ops.EagerTensor'>
'''

# tensor vs ndarray
c = np.array([[1,2],[3,4]])
d = tf.constant(c)
print(c)
print(d)
print(type(c))
print(type(d))
print("\n")
'''
[[1 2]
 [3 4]]
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int64)
<class 'numpy.ndarray'>
<class 'tensorflow.python.framework.ops.EagerTensor'>
'''

e = d.numpy()
print(e)
print(type(e))
print("\n")

'''
[[1 2]
 [3 4]]
<class 'numpy.ndarray'>
'''

# cpu or gpu
print(d.device)
print(tf.test.is_gpu_available())
print("\n")
'''
/job:localhost/replica:0/task:0/device:CPU:0
True
'''

# Variable
f = tf.Variable([[1,2],[3,4]])
print(f)
print(type(f))
print("\n")

'''
<tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=
array([[1, 2],
       [3, 4]], dtype=int32)>
<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
'''

# edit value of tensor
g = tf.constant(100)
print(g)
#g.assign(20)
#print(g)

'''
tf.Tensor(100, shape=(), dtype=int32)
Traceback (most recent call last):
  File "/home/peco/Desktop/Learn_TensorFlow2.0/01_tf_keras/tensor_and_basic_op/tensor.py", line 84, in <module>
    g.assign(20)
AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'assign'
'''

h = tf.Variable(100)
print(h)
h.assign(50)
print(h)
'''
tf.Tensor(100, shape=(), dtype=int32)
<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=100>
<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=50>
'''