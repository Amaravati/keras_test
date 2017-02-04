#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:23:34 2017

@author: anvesha
"""

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
    
def sigm(x,deriv=False):
    if(deriv==True):
        y=1/(1+np.exp(-x))
        return y*(1-y)
    return 1/(1+np.exp(-x))
    
def relu(x,deriv):
    if(deriv==False):
        return np.maximum(x,0)
    else:
        x[x>0]=1
        return x    
    

    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
alpha=0.01

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for j in range(20000):

    # forward propagation
    l0 = X
    l1 = relu(np.dot(l0,syn0),False)
    #print(l1)
    # how much did we miss?
    l1_error = y - l1
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l1_error))))
    
    l1_delta = l1_error * relu(np.dot(l0,syn0),True)

    # update weights
    syn0 += alpha*np.dot(l0.T,l1_delta)

print ("Output After Training:")
print (l1)