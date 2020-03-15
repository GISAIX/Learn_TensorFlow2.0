import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
# config GPU
# tf.debugging.set_log_device_placement(True)
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
print("num of physical gpus: ",len(phy_gpus))
for gpu in phy_gpus:
   tf.config.experimental.set_memory_growth(gpu,True)
 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.python.keras.optimizers import SGD

# 加载数据
def load_mnist_func(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(x_train_data,y_train_data),(x_test_data,y_test_data) = load_mnist_func(path='./data/mnist.npz')
print("x_train_shape:",x_train_data.shape)
print("y_train_shape:",y_train_data.shape)
print("x_test_shape:",x_test_data.shape)
print("x_test_shape:",y_test_data.shape)

# 处理数据
#60000x28x28 ==>60000x784 并且归一化
x_train_data = x_train_data.reshape(-1,28,28,1)/255.0

x_test_data = x_test_data.reshape(-1,28,28,1)/255.0

# ==>onehot
y_train_data = np_utils.to_categorical(y_train_data,num_classes=10)
y_test_data = np_utils.to_categorical(y_test_data,num_classes=10)

# 构建模型
model = Sequential()
model.add(Convolution2D(input_shape = (28,28,1),filters = 32,kernel_size = 3,
    strides = 1,padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2,strides =2,padding = 'same',))

model.add(Convolution2D(64,3,strides=2,padding='same',activation = 'relu'))
model.add(MaxPooling2D(2,2,'same'))
model.add(Convolution2D(128,3,strides=2,padding='same',activation = 'relu'))
model.add(MaxPooling2D(2,2,'same'))

#把第二个池化层的输出扁平化为1维
model.add(Flatten())
model.add(Dense(1024,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

# 定义优化器
sgd = SGD(lr=0.01)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train_data,y_train_data,batch_size=64,epochs=100)

# 评估模型
loss,accuracy = model.evaluate(x_test_data,y_test_data)

model_path="./weights/mnist_cnn.h5"
model.save_weights(model_path)

print('test loss',loss)
print('test accuracy',accuracy)