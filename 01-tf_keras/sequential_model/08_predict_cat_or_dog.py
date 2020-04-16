import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
# config GPU
# tf.debugging.set_log_device_placement(True)
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
print("num of physical gpus: ",len(phy_gpus))
for gpu in phy_gpus:
   tf.config.experimental.set_memory_growth(gpu,True)

from tensorflow.python.keras.preprocessing.image import *
from tensorflow.python.keras.models import load_model
import argparse as ap
import numpy as np
import cv2 as cv

parser = ap.ArgumentParser()
parser.add_argument("-i","--image_path",
    type = str,required=True)
args = parser.parse_args()

#加载模型
label = np.array(['猫','狗'])
model = load_model('01_tf_keras/sequential_model/weights/model_vgg16.h5')

def pred(img):
    image = load_img(img)
    cv.imshow("input",cv.cvtColor(np.asarray(image),cv.COLOR_RGB2BGR))
    image = image.resize((150,150))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,0)
    image.shape
    print(label[model.predict_classes(image)])
    cv.waitKey(0)

pred(args.image_path)