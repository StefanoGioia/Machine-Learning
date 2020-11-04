# -*- coding: utf-8 -*-
"""
Optimizing a model

@author: Stefano Gioia
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

#Insert path °#°#°#°#°#°######################################
setdir = r"C:\...\Images" #path where pickle data is #########

pickle_in = open(setdir+"\X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open(setdir+"\y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [0, 1, 2] #[2] 
layer_sizes = [32, 64, 128] #E.g. if better with 128, then try [64, 128, 256] #[64] 
conv_layers =  [1, 2, 3]  #[2] 

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            #Input layer
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            #Output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )
            
            y= np.array(y) #to use validation split

            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])

# 4 performances,type: tensorboard --logdir=logs/ from cmd.exe, after cd in folder
# model insights: model.summary()

#Save model
model.save('2-64-2-CNN.model')
