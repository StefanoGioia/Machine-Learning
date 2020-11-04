# -*- coding: utf-8 -*-
"""
Recurrent Neural Networks:  importance of sequential data!

@author: Stefano Gioia
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #, CuDNNLSTM if GPU (don't pass an activation, tanh is required)

#In this case digits are learnt by a sequential read of image rows

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpack images 

# import numpy
# x = numpy.concatenate((x_train, x_test),axis=0)

x_train = x_train/255.0
x_test = x_test/255.0

# print(x_train.shape)    #disp size
# print(x_train[0].shape) #disp one element size

model = Sequential()
#Sequence of image rows (regular NN): 28 sequences of 28 elements

# LSTM: Long Short Term Memory for recurring data
# return_sequences: if false, nodes output only after all data processed
#If recurrent layer afterwards, return_sequences= True
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax')) #10 digits

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))

#test

predictions = model.predict(x_test)
print(predictions)
import numpy as np
sample=34
print(np.argmax(predictions[sample]),y_test[sample])
