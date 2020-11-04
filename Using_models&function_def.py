# -*- coding: utf-8 -*-
"""
Using models

@author: Stefano Gioia
"""

import cv2
import tensorflow as tf

Labels = ["Cat", "Dog"]

#Define function to prepare files to be "recognised"
def prepare(filepath):
    IMG_SIZE = 100  # the same size of training data! ################################
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) #read and greyscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) #shape for tf
    

# Insert data directory#############################################
datadir = r"C:\Users\..."

model = tf.keras.models.load_model("2-64-2-CNN.model") #Insert model to use ############

prediction = model.predict([prepare( datadir+"\10001.JPG")]) #list

print(prediction)  # list in a list
#Convert to int, so to the label
print(Labels[int(prediction[0][0])])