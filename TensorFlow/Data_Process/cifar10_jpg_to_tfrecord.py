import tensorflow as tf
import glob
import cv2
import numpy as np
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

idx = 0
im_data = []
im_labels = []
for class_ in classes:
    path = "train_data/cifar10_jpgs/" + class_
    im_list = glob.glob(path + "/*")
    im_label = [idx for i in  range(im_list.__len__())]
    idx += 1
    im_data += im_list
    im_labels += im_label

tfrecord_file = "train_data/train_data.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecord_file)

index = [i for i in range(im_data.__len__())]

np.random.shuffle(index)

for i in range(im_data.__len__()):
    im_d = im_data[index[i]]
    im_l = im_labels[index[i]]
    data = cv2.imread(im_d)
    ex = tf.train.Example(
        features = tf.train.Features(
            feature = {
                "image":tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[data.tobytes()])),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[im_l])),
            }
        )
    )
    writer.write(ex.SerializeToString())

writer.close()
print("make tfrecord_file done.")