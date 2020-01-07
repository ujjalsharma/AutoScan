from keras.preprocessing.image import ImageDataGenerator , img_to_array
import keras
import cv2 
import numpy as np
import tensorflow as tf
class edit:
	@staticmethod

	def aug(img , p, total):
		print ("Image Found: " + p)
		ext = p[p.rfind('.'):]
		aug = ImageDataGenerator()
		x_size = cv2.resize(img,(1000,1000)) 
		x = img_to_array (x_size)
		#rotations
		for types in range(0,6):
			p =  'C:\\Users\\Asus\\Project\\augdata\\'+ str(total+types).zfill(3) + ext
			if types == 0:
				rot1 = aug.apply_transform(x,{'theta':10})
				cv2.imwrite(p,rot1)
			elif types == 1:
				rot2 = aug.apply_transform(x,{'theta':-10})
				cv2.imwrite(p,rot2)
		#	flips
			elif types == 2:
				flip1 = aug.apply_transform(x,{'shear':20})
				cv2.imwrite(p,flip1)
			elif types == 3:
				flip2 = aug.apply_transform(x,{'flip_horizontal':1})
				cv2.imwrite(p,flip2)
			# brightness 
			elif types == 4:
				dark = aug.apply_transform(x,{'brightness':0.5})
				cv2.imwrite(p,dark)
			elif types == 5:
				light = aug.apply_transform(x,{'brightness':1.5})
				cv2.imwrite(p,light)
   