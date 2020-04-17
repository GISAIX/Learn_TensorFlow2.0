import os,sys
sys.path.append('models')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
# config GPU
# tf.debugging.set_log_device_placement(True)
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
print("num of physical gpus: ",len(phy_gpus))
for gpu in phy_gpus:
   tf.config.experimental.set_memory_growth(gpu,True)

# 导入模型
from vgg import vgg16

from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import *

weights_path = '01-tf_keras/sequential_model/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = vgg16(weights=weights_path,include_top=False,
                        input_shape=(150,150,3),classes=2)

model.add(Flatten(name='flatten'))
model.add(Dense(4096,activation='relu',name='fc_1'))
model.add(Dropout(0.3,name='dropout_1'))
model.add(Dense(4096,activation='relu',name='fc_2'))
model.add(Dropout(0.3,name='dropout_2'))
model.add(Dense(2,activation='softmax',name='predictions_layer'))
model.summary()

# 处理数据

train_datagen = ImageDataGenerator(
    rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2, rescale = 1/255,shear_range = 20,
    zoom_range = 0.2,horizontal_flip = True,fill_mode = 'nearest',) 
test_datagen = ImageDataGenerator(rescale = 1/255,) # 数据归一化 

batch_size = 32

# train_data
train_generator = train_datagen.flow_from_directory(
    '01-tf_keras/sequential_model/data/cat_vs_dog/train',
    target_size=(150,150),
    batch_size=batch_size)

# test_data
test_generator = test_datagen.flow_from_directory(
    '01-tf_keras/sequential_model/data/cat_vs_dog/test',
    target_size=(150,150),
    batch_size=batch_size )

# train
model.compile(optimizer=SGD(lr=0.001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator,steps_per_epoch=len(train_generator),
                    epochs=100,validation_data=test_generator,
                    validation_steps=len(test_generator))

model.save('01-tf_keras/sequential_model/weights/model_vgg16.h5',include_optimizer=True,save_format='h5')