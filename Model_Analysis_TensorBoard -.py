# -*- coding: utf-8 -*-
"""

Model analysis with TensorBoard

@author: Stefano Gioia
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


#A callback is an object that can perform actions (e.g.write logs) at various stages 
#of training (e.g. at the start or end of an epoch, before or after a single batch, etc).

#More info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard

import pickle
import time

###################################################
#Insert path ######################################
setdir = r"C:\Users\...\Images" #path where pickle data is

pickle_in = open(setdir+"\X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open(setdir+"\y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

#model ##

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu')) ##

model.add(Dense(1))
model.add(Activation('sigmoid'))

####

#Manke TensorBoard callback object:

log_name = "cats-vs-dogs-CNN"

tensorboard = TensorBoard(log_dir="logs/{}".format(log_name))
#save training data to logs/data_name (to be read by TensorBoard)

########


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

y= np.array(y) #to use validation split

model.fit(X, y,  batch_size=32, epochs=10, 
          validation_split=0.3,
    callbacks=[tensorboard])

#For performances, type: tensorboard --logdir=logs/ from cmd.exe, after cd in folder

#reminder: Convolution->grouping, Pooling: picking in subgroups
#Epoch: iteration over whole dataset
#Overfit -> "memorization" -> acc. up and loss too!