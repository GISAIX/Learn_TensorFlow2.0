#-*- coding:utf-8 -*-
'''VGG19 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

- https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py

'''

import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

# check version

for module in tf,np,keras:
    print(module.__name__,module.__version__)

# weights url

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG19(include_top = True,weights = 'imagenet',
          input_tensor = None,input_shape=None,pooling=None,
          classes = 1000):
    """ VGG19 Model.

    # Arguments
        include_top: 是否包含模型的3个全连接层
        weights: "None"(随机初始化)和"imagenet"(使用imagenet预训练的参数)中的一个
        input_tensor: 可选的来当做模型输入图像使用的Keras张量(也就是"layers.Input()"的输出)
        input_shape: 可选的形状元组, 只有在"include_top"为False的时候可以用来指定
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: 类别数量

    # Returns
        A Keras model instance.

    """

    # 检查weights参数输入是否合法
    if weights not in {'imagenet',None}:
        raise ValueError('The `weights` argument should be either `None`  or `imagenet`.')

    if weights == 'imagenet' and include_top and classes!=1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = keras.layers.Input(shape = input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor = input_tensor,shape = input_shape)
        else:
            img_input = input_tensor


    # use functional api

    # block 1
    net = Conv2D(64,(3,3),activation ='relu',padding='same',name='block1_conv1')(img_input)
    net = Conv2D(64,(3,3),activation ='relu',padding='same',name='block1_conv2')(net)
    net = MaxPooling2D((2,2),stides=(2,2),name='block1_pool')(net)

    # block 2
    net = Conv2D(128,(3,3),activation ='relu',padding='same',name='block2_conv1')(net)
    net = Conv2D(128,(3,3),activation ='relu',padding='same',name='block2_conv2')(net)
    net = MaxPooling2D((2,2),stides=(2,2),name='block2_pool')(net)

    # block 3
    net = Conv2D(256,(3,3),activation ='relu',padding='same',name='block3_conv1')(net)
    net = Conv2D(256,(3,3),activation ='relu',padding='same',name='block3_conv2')(net)
    net = Conv2D(256,(3,3),activation ='relu',padding='same',name='block3_conv3')(net)
    net = Conv2D(256,(3,3),activation ='relu',padding='same',name='block3_conv4')(net)
    net = MaxPooling2D((2,2),stides=(2,2),name='block3_pool')(net)

    # block 4
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block4_conv1')(net)
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block4_conv2')(net)
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block4_conv3')(net)
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block4_conv4')(net)
    net = MaxPooling2D((2,2),stides=(2,2),name='block4_pool')(net)

    # block 5
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block5_conv1')(net)
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block5_conv2')(net)
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block5_conv3')(net)
    net = Conv2D(512,(3,3),activation ='relu',padding='same',name='block5_conv4')(net)
    net = MaxPooling2D((2,2),stides=(2,2),name='block5_pool')(net)

    if include_top:
        net = Flatten(name='flatten')(net)
        net = Dense(4096,activation='relu',name='fc1')(net)
        net = Dropout(0.5)(net)
        net = Dense(4096,activation='relu',name='fc2')(net)
        net = Dropout(0.5)(net)
        net = Dense(classes,activation='softmax',name='preds')(net)
    else:
        if pooling == 'avg':
            net = keras.layers.GlobalAveragePooling2D()(net)
        elif pooling == 'max':
            net = keras.layers.GlobalMaxPooling2D()(net)

    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # functional api 搭建模型始于Input终于Model
    model = keras.Model(inputs,net,name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)

        if keras.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
    return model