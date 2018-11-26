# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:31:38 2018

@author: esteban struve

PCA Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import torchvision.datasets as datasets

def plot_images(dat, k=20, size=28):
    """ Plot the first k vectors as 28 x 28 images """
    x2 = dat[0:k,:].reshape(-1, size, size)
    x2 = x2.transpose(1, 0, 2)
    fig, ax = plt.subplots(figsize=(20,12))
    ax.imshow(x2.reshape(size, -1), cmap='bone')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# Load full dataset
X = mnist_trainset.train_data.numpy().reshape((60000, 28*28))
y = mnist_trainset.train_labels.numpy()

# Take out random subset
rp = np.random.permutation(len(y))
X = X[rp[:5000], :]
y = y[rp[:5000]]

### YOUR CODE HERE
pca = PCA(n_components=2)
pca.fit(X)
### END CODE
