# Importing the important libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import CharRecognition

# Input image dimensions
IMG_WIDTH = 75
IMG_HEIGHT = 100
IMG_CHANNEL = 1

#Load data
(train_images, train_labels), (test_images, test_labels) = CharRecognition.LoadData('letter_dataset', IMG_HEIGHT, IMG_WIDTH)

# reshape data
train_images, test_images = CharRecognition.ReshapeData(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL, train_images, test_images)

# Building model
model = CharRecognition.GenerateModel(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)

# Model Summary
print(model.summary())

#Fitting data
model.fit(train_images, train_labels, epochs=10)

# Evaluating testing data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# print the test loss and accuracy
print("Test loss: "+str(test_loss))
print("Test accuracy: "+str(test_acc))

# exporting the classified
model.save('cnn_char_classifier_01.h5')

