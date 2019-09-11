#1.导入相应的包
import keras as k
import numpy as np
import matplotlib.pyplot as plt
#keras按顺序构成的模型
from keras.models import Sequential
#DenseQ全连接层
from keras.layers import Dense

#2.生成实验数据
#生成100个随机点
x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)
y_data = x_data*0.1+0.2+noise

#显示随机点
plt.scatter(x_data,y_data)
plt.show()

#3.构建模型
model = Sequential()
#units输出维度 input_dim输入维度
model.add(Dense(units=1,input_dim=1))
model.compile(optimizer='sgd',loss='mse')

#train
for step in range(10001):
    cost = model.train_on_batch(x_data,y_data)
    if step%1000 ==0:
        print('cost:',cost)
        
W,b=model.layers[0].get_weights()
print('W:',W,'b:',b)

#预测值
y_pred = model.predict(x_data)
plt.scatter(x_data,y_data)
plt. plot(x_data,y_pred,'r-',lw=3)
plt.show()