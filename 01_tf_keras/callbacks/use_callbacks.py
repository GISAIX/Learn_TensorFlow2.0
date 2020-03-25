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
from alexnet import alexnet
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *

base_model = alexnet(weights=None,
                include_top=False,input_shape=(227,227,3),classes=2)
# 添加全局平均池化层
net = base_model.output
net = GlobalAveragePooling2D()(net)

# 添加全连接层
net = Dense(256, activation='relu',name='fc')(net)
net = Dropout(0.3)(net)
predictions = Dense(2, activation='softmax',name='predictions_layer')(net)

# 构建完整模型
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# 处理数据
train_datagen = ImageDataGenerator(
    rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2, rescale = 1/255,shear_range = 20,
    zoom_range = 0.2,horizontal_flip = True,fill_mode = 'nearest',) 
test_datagen = ImageDataGenerator(rescale = 1/255,) # 数据归一化 

batch_size = 16

# train_data
train_generator = train_datagen.flow_from_directory(
    '01_tf_keras/sequential_model/data/cat_vs_dog/train',
    target_size=(227,227),
    batch_size=batch_size)

# test_data
test_generator = test_datagen.flow_from_directory(
    '01_tf_keras/sequential_model/data/cat_vs_dog/test',
    target_size=(227,227),
    batch_size=batch_size )

# callbacks
# Tensorboard
log_dir = './tensorboard'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb_callback = TensorBoard(log_dir=log_dir,update_freq='batch')

# 保存最优模型权重
model_name='alexnet_model.h5'
save_best_model_callback = ModelCheckpoint(model_name,save_best_only=True)

# early stopping
early_stopping_callback = EarlyStopping(patience=5,min_delta=1e-3)

# 动态衰减学习率
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.001)
callbacks = [tb_callback,save_best_model_callback,early_stopping_callback,reduce_lr_callback]


# train
model.compile(optimizer=SGD(lr=0.001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator,steps_per_epoch=len(train_generator)
                    ,epochs=1000,validation_data=test_generator,
                    validation_steps=len(test_generator),
                    callbacks=callbacks)