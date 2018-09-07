# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:34:34 2018

@author: esteban struve
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
#from sklearn.tree import DecisionTreeClassifier

def get_2D_data():
    """ Generate a data set to play with"""
    X, y = make_classification(n_samples=500, n_features=2,
                               n_redundant=0, n_informative=2,
                               random_state=0, n_clusters_per_class=1)
    return X, y

def visualize2D(classifier, X, y):
    """ Visualize a 2D classifer on data"""
    h = 0.2
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    ax.contour(xx, yy, Z, [0], colors='k', linewidths=3)

    ax.scatter(X[:, 0], X[:, 1], c=1-y, cmap=plt.cm.Paired,
               edgecolors='black', s=25)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('Decision Boundary and Contour Plot - the stronger the color the more certain the prediction. \
                 The black line indicated the data points where the model think the label of the class is 50/50 percent')
    plt.show()
    
def simple2D_logreg(feat, labels):
    """ Run a simple (linear) classification model and see the results

    Step 1. Create LogisticRegression classifier object
    Step 2. Fit the data using the fit method
    Step 3. Print the result (use score function on classfier ) on the training data
    Step 4. Return the classifier object
    """
    # knock yourself out - pass means do nothing
    ### YOUR CODE 3 lines
    l = LogisticRegression()
    l.fit(feat,labels)
    print(l.score)
    return l
    ### END CODE

feat, labels = get_2D_data()
logistic = simple2D_logreg(feat, labels)
visualize2D(logistic, feat, labels)
plt.show()