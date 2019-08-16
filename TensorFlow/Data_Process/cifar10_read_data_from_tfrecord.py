import tensorflow as tf
import numpy as np
from cv2 import cv2 as cv
import glob
import os


filelist = ['train_data.tfrecord']
file_queue = tf.train.string_input_producer(filelist,num_epochs=None,shuffle=True)
reader = tf.TFRecordReader()
 _,ex = reader.read(file_queue)

batch = tf.train.shuffle_batch([ex],batch_size,
                                    capacity=batch_size*10,
                                    min_after_dequeue=batch_size*5)
example = tf.parse_example(batch,features=features)

image = example['image']
label = example['label']

image = tf.decode_raw(image,tf.uint8)
image = tf.reshape(image,[-1,32,32,3])

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    for i in range(5):
        print(i)
        image_bth,image_lab = sess.run([image,label])

        cv.imshow(str(image_lab),image_bth[0,...])
        cv.waitKey(0)