# -*- coding: utf-8 -*-
"""
NN for number identification from image 

@author: Stefano Gioia
"""

import tensorflow.keras as keras

#Check tensorflow version
import tensorflow as tf
print(tf.__version__)

#Hand written digits dataset, in mnist
#mnist.load_data() returns tuple of Numpy arrays
mnist = tf.keras.datasets.mnist

#extract data to train and test the network
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#size of array
x_train.shape

#visualize image
import matplotlib.pyplot as plt

plt.imshow(x_train[0])
#white and black
plt.imshow(x_train[0],cmap=plt.cm.binary)
#plt.show()

#plt.close('all')

#normalize data between 0 and 1, makes things easier
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0],cmap=plt.cm.binary)

## Feed-forward (vs recurrent) model , asid lllayers:max min dist each non out node
model = tf.keras.models.Sequential()

##LAYERS

#flat input layer: "nodes in vector" instead of array (multi-dim. layer) 
model.add(tf.keras.layers.Flatten()) #image reshaped to vector by flatten:28*28=784

#hidden layers; dense: fully connected, 128 nodes
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
#nn.relu Applies the rectified linear unit function element-wise, works well

#another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#output layer, 10 digits so 10 nodes
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #softmax:probabilities rather like y/n
#check softmax ###############################


# (x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9
# x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
# x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

# model = tf.keras.models.Sequential()
# #model.add(tf.keras.layers.Flatten())   #Flatten the images! Could be done with numpy reshape
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))   #10 because dataset is numbers from 0 - 9

#settings for model optimization so training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#features which works well (loss is kind of error)

##          TRAIN         ## 
model.fit(x_train, y_train, epochs=3) #epochs: times data go through network...too many times overfitting

#"quality" of obtained model   from train/in-sample data
val_loss_train, val_acc_train = model.evaluate(x_train, y_train)
print(val_loss_train)
print(val_acc_train)
  
#"quality" of obtained model   from test/out-sample data
val_loss, val_acc = model.evaluate(x_test, y_test) 
print(val_loss)
print(val_acc)  #if quality equal or better than train, it could be not a good thing!!!!!!


#save model
model.save('numbers_reader.model')
  
#load model
# new_model = tf.keras.models.load_model('epic_num_reader.model')

#Check what trained network returns, in probabilistic terms!
predictions = model.predict(x_test)
print(predictions)

#number recognised by the nn (outp. node with highest probability)
import numpy as np
print(np.argmax(predictions[3]))

#what you recognise is
plt.imshow(x_test[3],cmap=plt.cm.binary)

#ranking of most probable values for image 4
pred3=predictions[3]
print('Most probable values ranking: ')
print(np.argsort(-pred3))

#corresponding probabilities
print('with probabilities: %d', -np.sort(-pred3))

#or
# sorted(pred3,reverse=True)
# pred3[0]