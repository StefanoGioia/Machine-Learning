# -*- coding: utf-8 -*-
"""
Convolutional Neural Network (Convnets and CNNs)

@author: Stefano Gioia
"""
## CNN intuition:
# Convolution("sample areas", feature map) -> Pooling(give a value to the area)-> Convolution -> 
# -> Pooling -> Fully Connected Layer (typical NN) -> Output
#Each convolution and pooling step is a hidden layer

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential #to write less
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

##LOAD DATA
import pickle
dir = r"C:\Users\...\datasets\PetImages" #data folder
pickle_in = open(dir+"\X.pickle","rb")
X = pickle.load(pickle_in)
X = X/255.0 #Normalize 

X_test = X[520:563]
X = X[0:520]

pickle_in = open(dir+"\y.pickle","rb")
y = pickle.load(pickle_in)
y_test = y[520:563]
y = y[0:520]


##DEFINE (Sequential) MODEL AND ADD LAYERS
model = Sequential()

#256 input nodes
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:])) #1st layer:put input shape
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1)) 
model.add(Activation('sigmoid')) #Final output

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

y= np.array(y) #to use validation split

model.fit(X, y, batch_size=25, epochs=3, validation_split=0.3) 
#batch_size: batch of samples to propagate until set completed; reduces memory usage

#save model
model.save('Cat&Dog.model')
  
## CHECK MODEL
# import tensorflow as tf
# model = tf.keras.models.load_model('Cat&Dog.model')
# ##LOAD DATA
# import pickle
# dir = r"...\datasets\PetImages" 
# pickle_in = open(dir+"\X.pickle","rb")
# X = pickle.load(pickle_in)
# X = X/255.0 #Normalize 

# X_test = X[520:563]
# X = X[0:520]

# pickle_in = open(dir+"\y.pickle","rb")
# y = pickle.load(pickle_in)
# y_test = y[520:563]
# y = y[0:520]

predictions = model.predict(X_test)
predictions = np.rint(predictions)

y_test= np.array(y_test) #to use validation split
y_test = y_test.reshape(42,1)
compare = np.array([predictions, y_test] )
print(np.transpose(compare)) #concatenate column-wise
print(np.absolute(predictions-y_test))
err = 100*np.sum(np.absolute(predictions-y_test))/len(y_test)
print('error: ' + str(err) +' %')

loss, acc=model.evaluate(X_test,y_test)