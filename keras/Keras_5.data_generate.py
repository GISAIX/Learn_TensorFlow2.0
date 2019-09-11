import numpy as np
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

'''
ImageDataGenerator参数：
* rotation_range是一个0~180的度数，用来指定随机选择图片的角度。  
* width_shift和height_shift用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比  
* rescale值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。  
* shear_range是用来进行错切变换的程度，参考错切变换 
* zoom_range用来进行随机的放大  
* horizontal_flip随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候  
* fill_mode用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素
'''

datagen = ImageDataGenerator(
    rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2,rescale = 1/255,
    shear_range = 20,zoom_range = 0.2,horizontal_flip = True,fill_mode = 'nearest') 

img = load_img('Keras\data\catVSdog/train\cat/cat.1.jpg')
x = img_to_array(img)
x = np.expand_dims(x,0)

# 生成20张图片
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='Keras\data\genreate_images', save_prefix='new_cat', save_format='jpeg'):
    i += 1
    if i==20:
        break
print('image generate finished!')