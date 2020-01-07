from Augpy.daug import edit
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import os
from imutils import paths
import random 

p = sorted(list(paths.list_images("C:\\Users\\Asus\\Project\\Dataset\\")))
count =0
#image = cv2.imread(p)
random.seed(42)
random.shuffle(p)
for imgpath in p:
	image = cv2.imread(imgpath)
	edit.aug(img = image, p = imgpath,total = count)
	count +=6
print("Augmentation on selected folder done")
#print(image)
#edit.aug(img = image , p = p,total = count)
