#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:27:59 2017

@author: anvesha
"""

import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
 
def tanf(x,deriv=False):
	if(deriv==True):
         #y=np.tanh(x) 
	    return (1-np.power(np.tanh(x),2))

	return np.tanh(x) 
 
    
def relu(x,deriv):
    if(deriv==False):
        return np.maximum(x,0)
    else:
        x[x>0]=1
        return x
         
    
    
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[1],
			[1],
			[0],
			[1]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
alpha=0.001

for j in range(160000):

	# Feed forward through layers 0, 1, and 2
    l0 = X

    l1 = tanf(np.dot(l0,syn0),False)
    l2 = tanf(np.dot(l1,syn1),False)

    # how much did we miss the target value?
    l2_error = y - l2
#    l2_error = l2 -y
    
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*tanf(np.dot(l1,syn1),True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * tanf(np.dot(l0,syn0),True)
#    l1_delta = l1_error * relu(l1,True)

    syn1 += alpha*l1.T.dot(l2_delta)
    syn0 += alpha*l0.T.dot(l1_delta)
    
    
print(l2)
#lf=relu(np.dot(l1,syn1),False)
#print(lf)