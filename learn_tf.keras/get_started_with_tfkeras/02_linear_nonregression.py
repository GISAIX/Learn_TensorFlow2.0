import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
# config GPU
# tf.debugging.set_log_device_placement(True)
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
print("num of physical gpus: ",len(phy_gpus))
for gpu in phy_gpus:
   tf.config.experimental.set_memory_growth(gpu,True)

import numpy as np
import matplotlib.pyplot as plt
# keras顺序构成模型
from tensorflow.python.keras.models import Sequential
# Dense全连接层
from tensorflow.python.keras.layers import Dense,Activation
# 导入SGD优化器
from tensorflow.python.keras.optimizers import SGD

# 利用numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise
plt.scatter(x_data,y_data)
plt.show()

# 显示随机点
plt.scatter(x_data,y_data)
plt.show()

'''
调整学习率的方法
默认lr=0.01，首先导入SGD：
from keras.optimizers import SGD
然后定义一个sgd：
sgd=SGD(lr=0.1）
'''
model = Sequential()
# 定义优化算法
sgd = SGD(lr=0.1)

#构建一个1-10-1结构的网络
model.add(Dense(units=10,input_dim=1))
model.add(Activation('tanh'))
model.add(Dense(units=1,input_dim=10))
model.add(Activation('tanh'))

# 编译模型，打印出模型结构
model.compile(optimizer=sgd,loss='mse')
model.summary()

for step in range(10001):
    cost=model.train_on_batch(x_data,y_data)
    if step%500==0:
        print("cost",cost)
              
y_pred = model.predict(x_data)
plt.scatter(x_data,y_data)
plt. plot(x_data,y_pred,'r-',lw=3)
plt.show()