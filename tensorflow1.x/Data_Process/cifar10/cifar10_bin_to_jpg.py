import os
import pickle
import numpy as np
from cv2 import cv2 as cv

classes= ['airplane','automobile','bird','cat',
            'deer','dog','frog','horse','ship','truck']

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data  = []
labels = []
for i in range(1,6):
    file_name = "./cifar-10-python/data_batch_"+str(i)
    result = unpickle(file_name)
    data += list(result[b"data"])
    labels += list(result[b"labels"])
    print(file_name+" loaded.")
    
imgs = np.reshape(data, [-1, 3, 32, 32])

for i in range(imgs.shape[0]):
    print("processing image {}.".format(i))
    im_data = imgs[i, ...]
    im_data = np.transpose(im_data, [1, 2, 0])
    im_data = cv.cvtColor(im_data, cv.COLOR_RGB2BGR)
    
    f = r"{}/{}".format("./Data_Process/images", classes[labels[i]])

    if not os.path.exists(f):
        os.makedirs(f)

    cv.imwrite("{}/{}.jpg".format(f, str(i)), im_data)

print("All Done.")