from cv2 import cv2 as cv
import tensorflow as tf 
import numpy as np
import os

saved_path = r'TensorFlow\Data_Process\catVSdog\catdog_train.tfrecord'

file_list = os.listdir(r"TensorFlow\Data_Process\catVSdog\train")
np.random.shuffle(file_list)

"""
img = cv.imread(r"TensorFlow\Data_Process\catVSdog\train/"+file_list[1])
img = cv.resize(img,(227,227))
cv.imshow("aaa",img)
cv.waitKey(0)
"""

writer = tf.python_io.TFRecordWriter(saved_path)

lab = 0
for file in file_list:
    img = cv.imread(r"TensorFlow\Data_Process\catVSdog\train/"+file)
    img = cv.resize(img,(227,227))

    # cat : 0 dog : 1
    if file[0] == 'c':
        lab = int(0)
    elif file[0] == 'd':
        lab = int(1)
    ex = tf.train.Example(
        features = tf.train.Features(
            feature={
                "image":tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[img.tostring()])),                  
                "label":tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=[lab]))
            }          
       )
    )
    writer.write(ex.SerializeToString())
writer.close()

print("All Done!")