import os
import pickle
import numpy as np
import tensorflow as tf
from cv2 import cv2 as cv

# bin to tfrecord

classes= ['airplane','automobile','bird','cat',
            'deer','dog','frog','horse','ship','truck']

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data  = []
labels = []

file_name = r"TensorFlow\Data_Process\cifar-10-python\test_batch"
saved_path=r'TensorFlow\Data_Process\test_data.tfrecord'
result = unpickle(file_name)

data += list(result[b"data"])
labels += list(result[b"labels"])
print(file_name+" loaded.")

imgs = np.reshape(data, [-1, 3, 32, 32])
print(imgs.shape[0])
print(len(labels))

writer = tf.python_io.TFRecordWriter(saved_path)
for i in range(imgs.shape[0]):
    print(i)
    im_d = np.transpose(imgs[i],[1,2,0]) #original shape of imgs[0] is 3x32x32
    im_d = cv.cvtColor(im_d,cv.COLOR_RGB2BGR)
    im_l = labels[i]
    ex = tf.train.Example(
    features = tf.train.Features(
        feature = {
            "image":tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[im_d.tobytes()])),
            "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[im_l])),
                    }
        )
    )
    writer.write(ex.SerializeToString())

writer.close()
print("Generate Test Tfrecord Done.")