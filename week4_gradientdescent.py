# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:49:46 2018

@author: esteban struve
"""

import numpy as np
import matplotlib.pyplot as plt

def f(a, x):
    return 0.5 * (x[0]**2 + a * x[1]**2)

def visualize(a, path, ax=None):
    """
    Make contour plot of f_a and plot the path on top of it
    """
    x = np.arange(-257, 257, 0.1)
    y = np.arange(-100, 100, 0.1)
    xx, yy = np.meshgrid(x, y)
    z = 0.5 * (xx**2 + a * yy**2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 13))
    h = ax.contourf(xx, yy, z, cmap=plt.get_cmap('jet'))
    ax.plot([x[0] for x in path], [x[1] for x in path], 'w.--', markersize=4)
    ax.plot([0], [0], 'rs', markersize=8) # optimal solution
    ax.set_xlim([-257, 257])
    ax.set_ylim([-100, 100])

def gd(a, step_size=0.1, steps=42):
    """ Run Gradient descent
        fill list out with the sequence of points considered during the descent.         
    """
    out = []
    ### YOUR CODE HERE
    out.append(np.array(256,1))
    for i in range(steps):
        point = out[i]
        gradient = (0.5*2*a[i],0.5*2*a[i+1])
        npoint = point - step_size*gradient
		out.append(npoint)
    ### END CODE
    return out

fig, axes = plt.subplots(2, 3, figsize=(20, 16))
ateam = [[1, 4, 16], [64, 128, 256]]
for i in range(2):
    for j in range(3):
        ax = axes[i][j]
        a = ateam[i][j]
        path = gd(a) # use good step size here
        visualize(a, path, ax)
        ax.set_title('gd a={0}'.format(a))