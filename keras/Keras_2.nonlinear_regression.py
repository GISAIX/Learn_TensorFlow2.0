#1.导入相应的包
import keras as k
import numpy as np
import matplotlib.pyplot as plt
#keras按顺序构成的模型
from keras.models import Sequential
#DenseQ全连接层
from keras.layers import Dense,Activation
from keras.optimizers import SGD

#2.生成实验数据
x_data = np.linspace(-0.5,0.5,200)
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise
plt.scatter(x_data,y_data)
plt.show()

'''
调整学习率的方法
默认lr=0.01，首先导入SGD：
from keras.optimizers import SGD
然后定义一个sgd：
sgd=SGD(lr=0.1）
'''

#定义模型
model = Sequential()
#定义优化算法
sgd = SGD(lr=0.1)

#构建一个1-10-1结构的网络
model.add(Dense(units=10,input_dim=1))
model.add(Activation('tanh'))
model.add(Dense(units=1,input_dim=10))
model.add(Activation('tanh'))

model.compile(optimizer=sgd,loss='mse')
for step in range(10001):
    cost=model.train_on_batch(x_data,y_data)
    if step%500==0:
        print("cost",cost)
              
y_pred = model.predict(x_data)
plt.scatter(x_data,y_data)
plt. plot(x_data,y_pred,'r-',lw=3)
plt.show()