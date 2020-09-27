"""
Linear Regression

@author: Stefano Gioia
"""
#ML repository UCI

#clean "all": only deletes names that do not start with an underscore (pickle doesn't work)
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#for plotting session
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("...\student-mat.csv", sep=";")

print(data.head()) #head: first 5 elements
data= data[["G1","G2","G3","studytime","absences","failures"]]
print(data.head())

predict = "G3"   #label for G3, to get (g1,g2 atc are attributes)    

x= np.array(data.drop([predict],1)) #data without G3
y= np.array(data[predict]) #takes only g3

#split data above in these section

#CAREFUL WHEN RE-USING VARIBLES (NAMES)

#comment: use ''' #before begin and after the end
#now, keep trying fitting till desired accuracy

#find and save best case in 30 iterations
best = 0
for _ in  range(30): #range : from 1 to

    #DECLARE arrays in this loop for function sklearn otherwise doesn't work
    x = np.array(data.drop([predict], 1))  # data without "predict"
    y = np.array(data[predict])  # takes only "predict"
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # test_size proportion for data

    linear = linear_model.LinearRegression()  # labeling

    linear.fit(x_train, y_train)  # call function from labeled

    acc = linear.score(x_test, y_test)  # check how well, ratio on 1

    print(acc)

    if acc> best : #if better, it saves
        #save
        with open("saved_studmodel.pickle", "wb") as f:  # wb writes if not exist
            pickle.dump(linear, f)  # linear in this f file
        best=acc

    print('best acc is: ', best)
    # Now, predict for test values
    predictions = linear.predict(x_test)  # so that can check with y_test

    #show comparison prediction
    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])
        print(predictions[x] - y_test[x])  # just chec difference with predicted and actual

    # show a, b coefficient of linear regression for the best found case
    print('Coefficients: \n', linear.coef_)  # SLOPE,DEPENDENCE
    print('Intercept: \n', linear.intercept_)  # CONSTANT 
    '''
    # load saved
    pickle_in = open("saved_studmodel.pickle", "rb")
    linear = pickle.load(pickle_in)
    '''

#data to compare
p1= 'G1' #grade 1, can compare studytime and others
p2='G3'

style.use("ggplot")
#scatter plot
pyplot.scatter(data[p1], data[p2]) #not from x or y, original data
pyplot.xlabel(p1)
pyplot.ylabel("final grade") # that is G3
pyplot.show()

#iterations not always same result ("train" data change)
'''
#Here: only one iteration
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size =0.1) #test_size proportion for data

linear = linear_model.LinearRegression()  # labeling

linear.fit(x_train, y_train)  # call function from labeled
acc= linear.score(x_test,y_test)
print(acc)
#show a, b coefficient of linear regression
print('Coefficients: \n', linear.coef_) #CAN BE INTERPRETED EITHER AS SLOPE AND AS DEPENDENCE
print('Intercept: \n',linear.intercept_) # CONSTANT TERM OF LINE

#NOTE : RESULTS NOT SAME EVERY RUN 

#Now, predict for test values
predictions = linear.predict(x_test) #so that can check with y_test

for x in range(len(predictions)):

    print(predictions[x], x_test[x], y_test[x])
    print(predictions[x]-y_test[x])  #just check difference with predicted and act'ual

'''

