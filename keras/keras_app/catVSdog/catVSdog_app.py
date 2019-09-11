'''
usage:python catVSdog_app.py -i image_path
'''
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
import argparse as ap
import numpy as np
import cv2 as cv

parser = ap.ArgumentParser()
parser.add_argument("-i","--image_path",
    type = str,required=True)
args = parser.parse_args()

#加载模型
label = np.array(['猫','狗'])
model = load_model('models/model_vgg16.h5')

def pred(img):
    image = load_img(img)
    cv.imshow("input",cv.cvtColor(np.asarray(image),cv.COLOR_RGB2BGR))
    image = image.resize((150,150))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,0)
    image.shape
    print(label[model.predict_classes(image)])
    cv.waitKey(0)

pred(args.image_path)

