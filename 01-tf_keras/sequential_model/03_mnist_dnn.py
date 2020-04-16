import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import tensorflow as tf 
# config GPU
# tf.debugging.set_log_device_placement(True)
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
print("num of physical gpus: ",len(phy_gpus))
for gpu in phy_gpus:
   tf.config.experimental.set_memory_growth(gpu,True)

import numpy as np
from tensorflow.python.keras import Sequential

from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.regularizers import l2

def load_mnist_func(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(x_train_data,y_train_data),(x_test_data,y_test_data) = load_mnist_func(path='01_tf_keras/sequential_model/data/mnist.npz')

print("x_train_shape:",x_train_data.shape)
print("y_train_shape:",y_train_data.shape)
print("x_test_shape:",x_test_data.shape)
print("x_test_shape:",y_test_data.shape)

#60000x28x28 ==>60000x784 并且归一化
x_train_data = x_train_data.reshape(x_train_data.shape[0],-1)/255.0
x_test_data = x_test_data.reshape(x_test_data.shape[0],-1)/255.0

# ==>onehot
y_train_data = to_categorical(y_train_data,num_classes=10)
y_test_data = to_categorical(y_test_data,num_classes=10)

#model 784==>200==>100=>10
model = Sequential([
    Dense(units=200,input_dim=784,bias_initializer='one',kernel_regularizer=l2(0.0003),activation='tanh'),
    Dropout(0.4),
    Dense(units=100,input_dim=200,bias_initializer='one',kernel_regularizer=l2(0.0003),activation='tanh'),
    Dropout(0.4),
    Dense(units=10,input_dim=100,bias_initializer='one',kernel_regularizer=l2(0.0003),activation='softmax'),
])
sgd = SGD(lr=0.1)
model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#train
model.fit(x_train_data,y_train_data,batch_size=128,epochs=20)
model.save("01_tf_keras/sequential_model/weights/mnist_dnn.h5",
                include_optimizer=False,save_format='h5')

#evaluate of test data
loss,accuracy = model.evaluate(x_test_data,y_test_data)
print("test loss: ",loss)
print("test acc: ",accuracy)

#evaluate of train data
loss,accuracy = model.evaluate(x_train_data,y_train_data)
print("train loss:",loss)
print("train loss:",accuracy)