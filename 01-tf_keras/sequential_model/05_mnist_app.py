import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
# config GPU
# tf.debugging.set_log_device_placement(True)
phy_gpus = tf.config.experimental.list_physical_devices('GPU')
print("num of physical gpus: ",len(phy_gpus))
for gpu in phy_gpus:
   tf.config.experimental.set_memory_growth(gpu,True)

import cv2 as cv
import argparse as ap 
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("01-tf_keras/sequential_model/weights/mnist_cnn.h5")

def pred(file_name):
    img = cv.imread(file_name,0)
    #cv.imshow("input",img)
    img = cv.resize(img,(28,28),interpolation = 0)
    img = 255-img
    img = img.reshape(-1,28,28,1)/255.0
    result = model.predict(img)
    print("predict result:",np.argmax(result,axis=1))
    #cv.waitKey(0)
    return np.argmax(result,axis=1)

count = 0
right = 0
for img in os.listdir("01-tf_keras/sequential_model/data/test_numbers/"):
    label = img[0]
    result = pred("01-tf_keras/sequential_model/data/test_numbers/"+img)
    if str(result[0]) == label:
        right += 1
    count += 1

print("CNN accuracy：",right/count)

'''result:
predict result: [4]
predict result: [5]
predict result: [8]
predict result: [3]
predict result: [8]
predict result: [0]
predict result: [9]
predict result: [1]
predict result: [2]
predict result: [7]
CNN accuracy： 0.9
'''