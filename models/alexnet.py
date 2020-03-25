"""使用tf.keras functional api构建alexnet模型

# 引用和参考：
- [ImageNet classification with deep convolutional neural networks](
    https://www.onacademic.com/detail/journal_1000039913864210_2a08.html) 

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
import tensorflow as tf

import os

def alexnet(include_top=True,weights=None,
                    input_shape=None,pooling=None,classes=1000):
    """使用tf.keras functional api构建alexnet模型

    # Arguments
        include_top:是否包含网络最后的3层全连接层，默认为包含。
        weights:选择预训练权重，默认'None'为随机初始化权重。
        input_shape:输入的尺寸，应该是一个元组，当include_top设为True时默认为(224,224,3)，否则应当被
                    定制化，因为输入图像的尺寸会影响到全连接层参数的个数。
        pooling:指定池化方式。
        classes:类别数量。
    # Returns
        返回一个tf.keras model实例。
    # Raises
        ValueError：由于不合法的参数会导致相应的异常。
    """

    # 检测weights参数是否合法
    if weights != None and not os.path.exists(weights):
        raise ValueError("the input of weights is not valid")

    input_ = tf.keras.Input(shape=input_shape)
    
    # first layer
    net = Conv2D(96,11,strides=4,padding='valid',activation='relu',name='conv_1')(input_)
    net = BatchNormalization(axis=1)(net)
    net = MaxPooling2D(pool_size=3,strides=2,padding='same',name='maxpool_1')(net)

    # second layer
    net = Conv2D(256,5,strides=1,padding='same',activation='relu',name='conv_2')(net)
    net = BatchNormalization(axis=1)(net)
    net = MaxPooling2D(3,2,padding='valid',name='maxpool_2')(net)

    # third layer
    net = Conv2D(384,3,strides=1,padding='same',activation='relu',name='conv_3')(net)

    # forth and fifth layer
    net = Conv2D(384,3,strides=1,padding='same',activation='relu',name='conv_4')(net) 
    net = Conv2D(256,3,strides=1,padding='same',activation='relu',name='conv_5')(net)

    net = MaxPooling2D(3,2,padding='valid',name='maxpool3')(net)
    
    if include_top:
        net = Flatten(name='flatten')(net)
        net = Dense(4096, activation='relu', name='fc1')(net)
        net = Dropout(0.5,name='dropout_1')(net)
        net = Dense(4096, activation='relu', name='fc2')(net)
        net = Dropout(0.5,name='dropout_2')(net)
        net = Dense(classes, activation='softmax', name='predictions')(net)
    else:
        if pooling == 'avg':
            net = GlobalAveragePooling2D()(net)
        elif pooling == 'max':
            net = GlobalMaxPooling2D()(net)

    model = tf.keras.Model(input_, net, name='alexnet')

    # 加载权重
    if weights != None:
        model.load_weights(weights)
        print("Loading weigths from "+weights+" finished!")

    return model

if __name__=='__main__':
    
    # set gpu and env
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf 
    phy_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in phy_gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    
    # test
    model = alexnet(weights=None,input_shape=(227,227,3),include_top=True,classes=1000)
    model.summary()
    #   =================================================================
    #    Total params:  62,378,344
    #    Trainable params: 62,378,344
    #    Non-trainable params: 0