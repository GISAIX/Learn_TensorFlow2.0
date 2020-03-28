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
from tensorflow.python.keras.preprocessing.image import *

datagen = ImageDataGenerator(
    rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2,rescale = 1/255,
    shear_range = 20,zoom_range = 0.2,horizontal_flip = True,fill_mode = 'nearest') 

img = load_img('sequential_model/data/cat_vs_dog/train/cat/cat.1.jpg')
x = img_to_array(img)
x = np.expand_dims(x,0)

# 生成20张图片
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='sequential_model/data/generate_images', save_prefix='new_cat', save_format='jpeg'):
    i += 1
    if i==20:
        break
print('image generate finished!')