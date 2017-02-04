#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:19:27 2017

@author: anvesha
"""
import numpy as np

class neuralnet:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
#        
#        self.wih=2*np.random.random((self.hnodes,self.inodes)) - 1
#        self.who=2*np.random.random((self.onodes,1)) - 1
        self.wih=np.random.random((self.hnodes,self.inodes)) 
        self.who=np.random.random((self.onodes,1)) 
        #x=np.array([1,2,3])

        self.lr=learningrate
        
    def relu(self,x,deriv):
        self.x=x
        if(deriv==False):
            return np.maximum(self.x,0)
        else:
            self.x[self.x>0]=1
            return self.x
            
    def relu1(self,x,deriv):
#        self.x=x
        if(deriv==False):
            return np.maximum(x,0)
            
        else:
            x[x>0]=1
            return x       
         
        pass  
    
    def train(self,X,y):
        for j in range(100000):
            l0 = X
            l1 = self.relu1(np.dot(l0,self.wih),False)
            l2 = self.relu1(np.dot(l1,self.who),False)
#            print(l1)
#            print('\n')
#            print(l2)
            
            l2_error = l2-y
#            print ("Error:" + str(np.mean(np.abs(l2_error))))
            #print(l2_error)
            
            if (j% 10000) == 0:
                print ("Error:" + str(np.mean(np.abs(l2_error))))
                print(self.wih)
                print('\n')
                print(self.who)
                
                #print ("\n""MSE:" + str((l2_error**2).mean(axis=0)))
                
            l2_delta = l2_error*self.relu1(np.dot(l1,self.who),True)
            
            l1_error = l2_delta.dot(self.who.T)
            #print(l1_error)
            l1_delta = l1_error * self.relu1(np.dot(l0,self.wih),True)
            
            self.who += -self.lr*np.dot(l1.T,l2_delta)
            self.wih += -self.lr*np.dot(l0.T,l1_delta)
#            print('\n')
            
            
        return self.wih,self.who,l2
        pass
    
    
    def predict(self,X):
        l1 = self.relu1(np.dot(X,self.wih),False)
        l2 = self.relu1(np.dot(l1,self.who),False)
        return l2
        pass
    pass
      

    


def main():
    X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
    y = np.array([[0],
			[1],
			[1],
			[1]])
    a=neuralnet(4,3,4,0.01)
    [w1,w2,out]=a.train(X,y)
    #print(out)
    #print('\n')
    outf=a.predict(X)
    #print(outf)

    

if __name__ == "__main__":
    main()
    
    

    
    

        
    
    
    