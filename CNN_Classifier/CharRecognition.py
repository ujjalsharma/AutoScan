# Importing the important libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# Function to load data
def LoadData(folder_name, height, width):
        
        #Initialize the image and labels array
        images_arr = np.array([]).reshape(0,height,width)
        labels_arr = np.array([])
        
        #Defining a dictionary assigning number/label to each character
        dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
            11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
            21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
            30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

        #reversing the key_value 
        rev_dictionary = {v:k for (k,v) in dictionary.items()}

        #get list of characters folder containing the images
        char_dirs = [x[0] for x in os.walk(folder_name)][1:]
        
        #appending the letter dataset
        for char_dir in char_dirs:
                filelist = glob.glob(char_dir+'/*.jpg')
                sub_images = np.array([np.array(Image.open(fname)) for fname in filelist])
                sub_labels = [rev_dictionary[char_dir[-1:]]]*len(sub_images)
                images_arr = np.append(images_arr,sub_images, axis = 0)
                labels_arr = np.append(labels_arr,sub_labels, axis = 0)
                print("Loaded dataset for "+str(char_dir[-1:]))
        
        #Splitting train and test
        X_train, X_test, y_train, y_test = train_test_split(images_arr, labels_arr, test_size=0.2, random_state=42, shuffle=True)
        
        #return the train and test split
        return (X_train, y_train), (X_test, y_test)

#Function to reshape data
def ReshapeData(height, width, channel, train_images, test_images):
    train_images = train_images.reshape((train_images.shape[0], height, width, channel))
    test_images = test_images.reshape((test_images.shape[0], height, width,channel))
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, test_images


# Model Generation Function
def GenerateModel(height, width, channel):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
    model.add(layers.BatchNormalization(axis=channel))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization(axis=channel))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(35, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
