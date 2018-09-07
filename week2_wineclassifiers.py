# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:40:07 2018

@author: esteban struve
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.externals.six import StringIO  
from IPython.display import Image, display
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import train_test_split

def bar_plot_weights(weights, names):
    """ Bar plot of weights"""
    features = weights.shape[1]
    classes = weights.shape[0]
    fig, ax = plt.subplots(figsize=(13, 10))
    index = np.arange(features)
    bar_width = 0.15
    opacity = 0.8
    colors = ['r', 'g', 'b']

    for i in range(classes):
        rects2 = plt.bar(index + i * bar_width, weights[i, :], bar_width,
                     alpha=opacity,
                     label='Wine {0}'.format(i))
 
    ax.set_xlabel('feature')
    ax.set_ylabel('feature weight')
    ax.set_title('Wine Feature Weights')
    ax.set_xticks(index + bar_width / 3, names)
    ax.set_xticklabels(names)
    plt.legend()
 
    plt.tight_layout()
    plt.show()

def plot_tree(dtree, feature_names):
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    img = Image(graph.create_png())
    display(img)


def print_score(classifier, X_train, X_test, y_train, y_test):
    """ Simple print score function that prints train and test score of classifier - almost not worth it"""
    print('In Sample Score: ',
          classifier.score(X_train, y_train))
    print('Test Score: ',
          classifier.score(X_test, y_test))

    
def train_wine_logistic(X_train, y_train):
    """LogisticRegression as before -
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.decision_function
        return the trained logistic regression object
        
        For even this simple model there are many choices you can tune, 
        the most important parameter probably being  the C parameter which controls how "complex" we want the function. 
        The smaller C the simpler function. If you like try different value
        
        Step 1. Create LogisticRegression classifier object
        Step 2. Fit the data using the fit method
        Step 3. Return the Classifier
    """
    # Knock your self out
    ### YOUR CODE 3 lines
    l = LogisticRegression()
    l.fit(X_train,y_train)
    print(l.score)
    return l
    ### END CODE

    
def train_wine_dectree(X_train, y_train, max_depth=5):
    """
    DecisionTrees -
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    
    For this DecisionTree model there are even kmore knobs to turn. But for this exercise you must set the max_depth parameter and experiment with it a little. It will greatly affect how intricate a function you can learn.
    As talked about shortly, we often prefer simpler explanations if one exists so this is at least one way you can control this.
    
    The parameter max_depth can be set in the constructor for the tree object (as a named paramater) i.e. t = DecisionTreeClassifier(max_depth=42)
    
    Remember to return the tree.
    
        Step 1. Create DecisionTreeClassifier object
        Step 2. Fit the data using the fit method
        Step 3. Return the classifier 
    """
    # knock yourself out
    ### YOUR CODE 3 lines
    d = DecisionTreeClassifier(max_depth=max_depth)
    d.fit(X_train, y_train)
    return d
    ### END CODE

       
def train_wine_neural_net(X_train, y_train, hidden_layers=(32, 32, 32)):
    """
   Neural Nets -
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    Remember to set the hidden_layer_sizes in the construction i.e. MLPClassifier(hidden_layers_sizes=)
    Otherwise train the MLPclassifer and return it/
    
    May be hard to get good results out of the box! Dont lose to much sleep if the results are not to good.
    Setting other hyperparameters away from standard is probably required... 
    Set batch_size to a small number like 4 or 8 helps significantly
    
        Step 1. Create MPLClassifer  object
        Step 2. Fit the data using the fit method
        Step 3. Return the classifier 
    """
    # knock yourself out
    ### YOUR CODE 3 lines
    n = MLPClassifier(hidden_layers)
    n.fit(X_train, y_train)
    return n
    ### END CODE

# the main method here - comment out stuff you do not want to see
## Get Data
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


## Try logistic regression
head = '*'*20
print(head, 'Logistic Regression Wine Score', head)
logistic = train_wine_logistic(X_train, y_train)
print_score(logistic, X_train, X_test, y_train, y_test)
print('Logistic Regreesion Learned Weights: ')
bar_plot_weights(logistic.coef_, data.feature_names)
print(data.feature_names)

## try different max depth
print('\n', head,'Decision Tree Wine Score', head)
tree = train_wine_dectree(X_train, y_train, max_depth=5)
print_score(tree,  X_train, X_test, y_train, y_test)
print('Lets see the Tree')
#plot_tree(tree, data.feature_names)

## try different hidden layer settings, less is probably more
print('\n', head, 'Neural Net Wine Score:', head)
mlp = train_wine_neural_net(X_train, y_train, hidden_layers=(10,))
print_score(mlp, X_train, X_test, y_train, y_test)
