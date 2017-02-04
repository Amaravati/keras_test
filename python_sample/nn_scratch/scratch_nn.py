#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:27:59 2017

@author: anvesha
"""

import numpy as np

#def nonlin(x,deriv=False):
#	if(deriv==True):
#	    return x*(1-x)
#
#	return 1/(1+np.exp(-x))
# 
#def tanf(x,deriv=False):
#	if(deriv==True):
#         #y=np.tanh(x) 
#	    return (1-np.power(np.tanh(x),2))
#
#	return np.tanh(x) 
# 
#    
#def relu(x,deriv):
#    if(deriv==False):
#        return np.maximum(x,0)
#    else:
#        x[x>0]=1
#        return x
#         
    
class neuralnet:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        np.random.seed(1)
#        self.wih=2*np.random.random((self.hnodes,self.inodes)) - 1
#        self.who=2*np.random.random((self.onodes,1)) - 1

        self.wih=2*np.random.random((self.inodes,self.hnodes)) - 1
        self.who=2*np.random.random((self.hnodes,self.onodes)) - 1

        self.lr=learningrate
        
    def relu(self,x,deriv):
        if(deriv==False):
            return np.maximum(x,0)
        else:
            y=np.maximum(x,0)
            y[y>0]=1
#            x[x>0]=1
#            x[x<0]=0
            return y
         
        pass  

    def train(self,X,y):
        for j in range(60000):
            l0 = X
            l1 = self.relu(np.dot(l0,self.wih),False)
            l2 = self.relu(np.dot(l1,self.who),False)
            l2_error = y - l2
            
            
            if (j% 10000) == 0:
                print ("Error:" + str(np.mean(np.abs(l2_error))))
                print ("\n""MSE:" + str((l2_error**2).mean(axis=0)))
        
    # in what direction is the target value?
            l2_delta = l2_error*self.relu(np.dot(l1,self.who),True)
            
            l1_error = l2_delta.dot(self.who.T)
            
            l1_delta = l1_error * self.relu(np.dot(l0,self.wih),True)
    
            self.who += self.lr*np.dot(l1.T,l2_delta)
            self.wih += self.lr*np.dot(l0.T,l1_delta)
        
        return self.wih,self.who,l2
       
          
    
def main():
    a=neuralnet(3,5,1,0.01)
    X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
    
    y = np.array([[0],
			[1],
			[1],
			[0]])
    [w1,w2,out]=a.train(X,y)
    print(w2)
    print(w1)
    print(out)
    

if __name__ == "__main__":
    main()