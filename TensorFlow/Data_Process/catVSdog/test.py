from cv2 import cv2 as cv
import tensorflow as tf 
import numpy as np
import os

feature = {
    'image':tf.FixedLenFeature([],tf.string),
    'label':tf.FixedLenFeature([],tf.int64)
}

filelist = [r'C:\\Users\\Peco\\Desktop\\per_pro\\TensorFlow_and_Keras_notes\\TensorFlow\\Data_Process\\catVSdog\\catdog_train.tfrecord']
file_queue = tf.train.string_input_producer(filelist,num_epochs=None,shuffle=False)

reader = tf.TFRecordReader()
_,ex = reader.read(file_queue)

batchsize=3
batch = tf.train.shuffle_batch([ex],batchsize,capacity=batchsize*10,
                                    min_after_dequeue=batchsize*5)
example = tf.parse_example(batch,features=feature)

image = example['image']
label = example['label']

image = tf.decode_raw(image,tf.uint8)
image = tf.reshape(image,[-1,227,227,3])

with tf.Session() as sess:
    sess.run([tf.local_variables_initializer(),tf.global_variables_initializer()])
    coord=tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(10):
        print(i)
        image_bth,image_lab = sess.run([image,label])
        l=image_lab[0]
        lab = ['Cat','Dog']
        cv.imshow(lab[l],image_bth[0,...])
        cv.waitKey(0)
