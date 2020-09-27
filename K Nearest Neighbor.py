"""
K Nearest Neighbor 

@author: Stefano Gioia
"""
#Data: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
#first: panda defining attributes in car.data (first row, no spaces!)

#k : k closest data points vote on the category to assign a point; when odd no ties
#k too high: bias to most numerous categories
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("...\car.data" ) #data folder
print(data.head())

# non numerical features! no computations: convert in numerical: a preprocessing
le= preprocessing.LabelEncoder()
#take columns to transform (transformed automatically scale 0 to max (+1 increasing), recognises med low etc )
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#print(buying) to see conversion
predict ="class"

x= list(zip(buying,maint,doors,persons, lug_boot,safety)) #zip: packs/tuples objects
y= list(cls)

#train
# test_size proportion for data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  

#classifier, n_neighbors hyperparameter ("to set before") for class, instead of regression
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted= model.predict(x_test)

names = ["uancc","acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

# names vector element: names[predicted[x]]
