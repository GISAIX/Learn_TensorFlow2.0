from cv2 import cv2 as cv
import tensorflow as tf
import numpy as np
import glob
import os

batch_size = 1

classes = ['airplane','automobile','bird','cat',
            'deer','dog','frog','horse','ship','truck']

features = {
    'image':tf.FixedLenFeature([], tf.string),
    'label':tf.FixedLenFeature([], tf.int64)
}

#构建文件名队列
filelist = ["./train_data.tfrecord"]
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
        image_bth,image_lab = sess.run([image,label])
        print("image[{}],label:".format(i)+classes[image_lab[0]])
        cv.namedWindow(classes[image_lab[0]],cv.WINDOW_NORMAL)
        cv.imshow(classes[image_lab[0]],image_bth[0,...])
        cv.waitKey(0)