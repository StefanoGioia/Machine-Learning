# -*- coding: utf-8 -*-
"""
Logistic Regression

@author: Stefano Gioia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn #to check results with heatmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#import statistics #for adaptive step

########### 
#Data 
########### 

applicants = {'TestA': [78,75,69,71,68,73,69,72,74,69,61,69,71,68,77,61,58,65,54,59,62,60,55,55,57,67,66,58,65,66,64,62,66,66,68,65,67,58,59,69],
              'TestB': [8,7.8,6.6,7.4,7.8,7.4,4.6,6.6,6.6,3.4,5.4,7.4,7.4,6.6,6.6,6,5.4,7.4,5.4,4.6,6.6,4,4.6,5.4,6,6.6,7.4,4.6,7.4,6.6,6,5.4,8,6.6,6.6,4.6,5.4,6.6,3.4,7.4],
              'Experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'Selected': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

df = pd.DataFrame(applicants,columns= ['TestA', 'TestB','Experience','Selected'])
#df.head()

X = df[['TestA', 'TestB','Experience']]
y = df['Selected']  #labels

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0) # split memorized in random_state value

########### 
#w/o sklearn classifier: define functions
###########

def initializeWeights(X): #X: features
    m = X.shape[1] 
    w = np.zeros((1,m)) #0.5*(np.random.rand(1,m)-0.5) 
    b = 0 #0.5*(np.random.rand()-0.5)    
    return w,b

def sigmoid(log_odds):
    #"p"=sigmoid(z), z:log(odds)
    sigmOut = 1/(1+np.exp(-log_odds))
    return sigmOut

def prob_sigmoid(X,w,b):
    X_T  = X.T
    zOdds= np.dot(w,X_T)+b
    p_s  = sigmoid(zOdds) 
    return p_s

def model_result(w, b, X, Y):
    
    #Prepare data for cost and gradient
    predicted_prob = prob_sigmoid(X,w,b) 
        
    m = X.shape[0] #to scale/ avg cost
    
    # Cost = (-1/m)*LogLikelihood , so to minimize.(-1/m)* also to the gradient
    avg_cost = (-1/m)*(np.sum( (Y*np.log(predicted_prob)) + ((1-Y)*(np.log(1-predicted_prob)))))
    #Gradient 
    X_T = X.T #.T: for dot product (element-wise is default with np, even with .T)
    dw = (-1/m)*(np.dot(X_T, (Y -predicted_prob).T)) #dcost/dw
    db = (-1/m)*(np.sum(Y - predicted_prob))         #dcost/db
    
    avg_grad = {"dw": dw, "db": db} #dict object for easy output read
    return avg_grad, avg_cost

def optimize_model_weights(w, b, X, Y, step_size, iterations): #w weights, b constant term
    costs_seq = [] # To store cost sequence
    for i in range(iterations):
        
        dcost, cost = model_result(w,b,X,Y)
        
        dw = dcost["dw"]
        db = dcost["db"]
        
        step = step_size
        #step = step_size * abs(w.mean()+.1) #adaptive step
        
        w = w - step * (dw.T) #opposite direction to increase:gradient descent 
        b = b - step * db
    
        costs_seq.append(cost) 
    
    weights = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return weights, gradient, costs_seq

def predict_(prob_sigm):
    n = max(prob_sigm.shape)
    binary_pred = np.zeros(n)
    for i in range(n):
        if prob_sigm[0][i] >= 0.5:
            binary_pred[i] = 1
    return binary_pred.astype(int) #returns integer array

###########
#Compute
###########

def logreg(Xtrain,ytrain,iterations,stepsize):
    
    X_trainA = np.array(Xtrain)
    y_trainA = np.array(ytrain)
    
    w_,b_ = initializeWeights(X_trainA)
    w_train, grad_train, costs_train = optimize_model_weights(w_, b_, X_trainA, y_trainA, stepsize, iterations)
    return w_train, grad_train, costs_train

def checkprediction(Xtest,w_trained):
    
    X_testA       = np.array(X_test)
    probabilities = prob_sigmoid(X_testA, w_trained["w"], w_trained["b"])
    y_predicted   = predict_(probabilities)
    return y_predicted, probabilities


w_train, grad_train, costs_train = logreg(X_train,y_train,200,5e-4)
y_pred_nosklearn, prob_nosk= checkprediction(X_test,w_train)


print("No sklearn: ", y_pred_nosklearn)

########### 
## Use sklearn classifier 
###########

#Declare Logistic Regression
logreg_sk= LogisticRegression()

#Fit
logreg_sk.fit(X_train,y_train)

#Check results
y_pred=logreg_sk.predict(X_test)
print("sklearn:    ",y_pred)

print("True val:   ",np.array(y_test))

########### 
#Check results
########### 

confusion_mtx_nsk = pd.crosstab(y_test, y_pred_nosklearn, rownames=['Actual'], colnames=['Predicted_No_sklearn'])
sn.heatmap(confusion_mtx_nsk, annot=True)
plt.show()

print('Accuracy_No_sklearn: ',metrics.accuracy_score(y_test, y_pred_nosklearn))


confusion_mtx = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted_sklearn'])
sn.heatmap(confusion_mtx, annot=True)
plt.show()

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.plot(costs_train)
plt.xlabel("Epochs")
plt.ylabel("Cost")