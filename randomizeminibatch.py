# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:23:11 2018

@author: esteb
"""
import numpy as np
def iterate_minibatches(inputs, targets, batchsize):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    indices = np.random.shuffle(np.arange(inputs.shape[0]))
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
    return inputs[excerpt], targets[excerpt]

batch_size = 3
n_epochs = 1
Y = np.arange(36)
X = np.arange(36)
for n in range(n_epochs):
    for batch in iterate_minibatches(X, Y, batch_size):
        x_batch, y_batch = batch
        print(x_batch)