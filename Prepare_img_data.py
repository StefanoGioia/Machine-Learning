# -*- coding: utf-8 -*-
"""
Prepare data and save: make same size, label

@author: Stefano Gioia
"""

import numpy as np
import matplotlib.pyplot as plt
import os  #operating system dependent functionalities
import cv2  #computer vision problems, pip install opencv-python
from tqdm import tqdm #to display progress bars

setdir = r"C:\Users\...\datasets\PetImages"  #image data folder, r before to make valid

CATEGORIES = [ "Cat", "Dog"] #categories and folders of dataset

for category in CATEGORIES:  # for each category, i.e. dogs and cats
    path = os.path.join(setdir,category)  # create path to dogs and cats, in dir
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        #os.path.join(path,img) : image for cv2.imread
        
        ## CHECK AND CHOOSE SIZE FOR RESHAPING
        
        #show obtained image
        plt.imshow(img_array, cmap='gray')  # graph it
        #plt.show()  # display!

        # print(img_array) #values
        # print(img_array.shape) #size of the img
        
        #make 100x100 image
        IMG_SIZE =100
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        # plt.imshow(new_array, cmap='gray')
        #plt.show()
        
        break     #breaks , only a check at beginning
    break 

## prepare data

training_data = []

## FUNCTION ##
def create_training_data():
    for category in CATEGORIES: 
        print(len(training_data))
        path = os.path.join(setdir,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # classification,labels: 0=dog 1=cat
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            # print(img)
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                #plt.imshow(img_array, cmap='gray') #                 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            # except Exception as e:  # in the interest in keeping the output clean...
            #     pass
            # except OSError as e:
            #     print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print(img)
                print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

#Not balanced classes->the model will initially learn that the best thing to do is predict only one class
#minority class as rare case if highly unbalanced

## shuffle data: avoid predict always the same
import random
random.shuffle(training_data)
 #CHECK
for sample in training_data[:10]:
    print(sample[1]) # 1: prints only label not the array([0])
    
#prepare data for the model    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
#from list to array reshape(size)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #-1 take all arrays; 1: grayscale (1 color (unlike RGB))

## generate file to save with pickle

import pickle

pickle_out = open(setdir+"\X.pickle","wb")
pickle.dump(X, pickle_out) #variable to file opened by pickle_out
pickle_out.close()

pickle_out = open(setdir+"\y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# When data have to be loaded:
# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)









