#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:16:35 2017

@author: anvesha
"""

import numpy as np

x=np.array([0.5,-0.5])
w=np.array([2,-1])
b=0.5

y=np.dot(x,w.T)+0.5

xnew=x-w
print(y)
