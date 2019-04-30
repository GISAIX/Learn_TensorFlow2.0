import urllib
import os
import sys
import tarfile
import glob
import pickle
import numpy as np
import cv2

classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 解压一个二进制，解压后的数据以字典形式返回
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data  = []
labels = []
for i in range(1,6):
	file_name = "train_data/data_batch_"+str(i)
	result = unpickle(file_name)
	data += list(result[b"data"])
	labels += list(result[b"labels"])
	print(file_name+" loaded.")
	
imgs = np.reshape(data, [-1, 3, 32, 32])

for i in range(imgs.shape[0]):
    im_data = imgs[i, ...]
    im_data = np.transpose(im_data, [1, 2, 0])
    im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR)

    f = "{}/{}".format("train_data/jpgs", classes[labels[i]])

    if not os.path.exists(f):
        os.mkdir(f)

    cv2.imwrite("{}/{}.jpg".format(f, str(i)), im_data)

print("All Done.")












