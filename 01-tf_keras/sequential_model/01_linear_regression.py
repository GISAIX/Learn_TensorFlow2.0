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
# keras顺序构成的模型
from tensorflow.keras import Sequential
# Dense全连接层
from tensorflow.keras.layers import Dense

# 利用numpy生成100个随机点
x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)

# y=0.1*x+0.2
y_data = x_data*0.1+0.2+noise

# 显示随机点
plt.scatter(x_data,y_data)
plt.show()

# 构建模型
model = Sequential()
# units输出维度 input_dim输入维度
model.add(Dense(units=1,input_dim=1,name='dense_1'))
model.compile(optimizer='sgd',loss='mse')
model.summary()

# 训练模型
for step in range(10001):
    #print(step)
    cost = model.train_on_batch(x_data,y_data)
    if step%100 ==0:
        print('cost:',cost)
        
W,b=model.layers[0].get_weights()
print('W:',W,'b:',b)

#预测值
y_pred = model.predict(x_data)
plt.scatter(x_data,y_data)
plt. plot(x_data,y_pred,'r-',lw=3)
plt.show()