'''
usage:python mnist_cnn_app.py -i index.png
'''

import cv2 as cv
import matplotlib.pyplot as plt
import argparse as ap 
import numpy as np
from keras.models import load_model

parser = ap.ArgumentParser()
parser.add_argument("-i","--image_path",
    type = str,required=True)
args = parser.parse_args()

model = load_model("models\mnist_cnn.h5")

def pred(name):
    img = cv.imread(name,0)
    img = cv.resize(img,(28,28),interpolation = 0)
    img = 255-img
    plt.axis('off')
    plt.imshow(img)
    img = img.reshape(-1,28,28,1)/255.0
    result = model.predict(img)
    print("pre result:",np.argmax(result,axis=1))
    return np.argmax(result,axis=1)

pred(args.image_path)
