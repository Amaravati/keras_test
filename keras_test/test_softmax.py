#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:52:34 2017

@author: anvesha
"""
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(10, noise=0.20)
    return X, y


def predict(X):
    num_examples = len(X)
    nn_hdim=4
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
#    print(z2)
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z2,np.argmax(probs, axis=1)
    
    
    

def main():
    X, y = generate_data()
#    print(X)
    print(y)
    num_examples=len(X)
    [a,b]=predict(X)
    print(a)
    c=np.exp(a)
    print('\n')
    print(c)
    print('\n')
    d=c/np.sum(c,axis=1,keepdims=True)
    e=np.argmax(d,axis=1)
    print(d)
    print('\n')
    print(e)
    delta3 = d
    delta3[range(num_examples), y] -= 1
##    delta3=y-d
    print(delta3)


if __name__ == "__main__":
    main()
