#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:26:27 2017

@author: anvesha
"""
import numpy as np
from nn_sigmoid import neural_net, LossHistory

NUM_INPUT=2
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100


loss_log = []
nn_param=[20,50]
model = neural_net(NUM_INPUT, nn_param)
history = LossHistory()
model.fit(
          X, y, batch_size=3,
          nb_epoch=1, verbose=0, callbacks=[history]
)
loss_log.append(history.losses)


qval = model.predict(X, batch_size=1)

print(qval)
    
    
    
    
    