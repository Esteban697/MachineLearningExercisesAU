import numpy as np
from h1_util import numerical_grad_check

def softmax(X):
    """ 
    Compute the softmax of each row of an input matrix (2D numpy array).
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE no for loops please
    if X.ndim==1:
        maxrow = np.max(X)
        res1 = np.log(np.sum(np.exp(X - maxrow))) + maxrow
        res = np.exp(X - res1)
    else:
        idx=0
        for row in X:
            maxrow = np.max(row)
            res1 = np.sum(np.exp(row - maxrow))
            res2 = np.log(res1) + maxrow
            res3 = np.exp(row - res2)
            res[idx,:] = res3
            idx += 1
    ### END CODE
    return res

def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc
    
class SoftmaxClassifier():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = None
        
    def cost_grad(self, X, y, W):
        """ 
        Compute the average cross entropy cost and the gradient under the softmax model 
        using data X, Y and weight vector W.
        
        the functions np.nonzero, np.sum, np.dot (@), may come in handy
        Args:
           X: numpy array shape (n, d) float - the data each row is a data point
           y: numpy array shape (n, ) int - target values in 0,1,...,k-1
           W: numpy array shape (d x K) float - weight matrix
        Returns:
            totalcost: Average Negative Log Likelihood of w 
            gradient: The gradient of the average Negative Log Likelihood at w 
        """
        cost = np.nan
        grad = np.zeros(W.shape)*np.nan
        Yk = one_in_k_encoding(y, self.num_classes) # may help - otherwise you may remove it
        ### YOUR CODE HERE
        n = X.shape[0] #number of samples
        A = softmax(np.matmul(X,W))
        cost = -1/n * np.sum((Yk * np.log(A)))
        Xt = np.transpose(X)
        grad = np.matmul(-Xt, (Yk - A)) / n
        ### END CODE
        return cost, grad


    def fit(self, X, Y, W=None, lr=0.01, epochs=10, batch_size=16):
        """
        Run Mini-Batch Gradient Descent on data X,Y to minimize the in sample error (1/n)NLL for softmax regression.
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
           W: numpy array shape (d x K)
           lr: scalar - initial learning rate
           batchsize: scalar - size of mini-batch
           epochs: scalar - number of iterations through the data to use

        Sets: 
           W: numpy array shape (d, K) learned weight vector matrix  W
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        if W is None: W = np.zeros((X.shape[1], self.num_classes))
        history = []       
        ### YOUR CODE HERE 14 - 20 lines
        assert X.shape[0] == Y.shape[0] #check data
        for currentclass in range(self.num_classes):
            for epoch in range(epochs):
                indices = np.arange(X.shape[0])
                indices = np.random.permutation(indices) #Randomize training data using index
                for start_indx in range(0, X.shape[0] - batch_size + 1, batch_size):
                    chunk = indices[start_indx:start_indx + batch_size]
                    X_train_mini = X[chunk]
                    Y_train_mini = Y[chunk] #mini batch extracted
                    cost, gradient = self.cost_grad(X_train_mini,Y_train_mini,W)
                    W -= lr * gradient #change weight
                history.append(cost)
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lr, cost))
        ### END CODE
        self.W = W
        self.history = history
        

    def score(self, X, Y):
        """ Compute accuracy of classifier on data X with labels Y

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
        Returns:
           out: float - accuracy
        """
        out = 0
        ### YOUR CODE HERE 1-4 lines
        guess = self.predict(X)
        errors = guess == Y #check data
        out =  np.count_nonzero(errors)/X.shape[0]
        ### END CODE
        return out

    def predict(self, X):
        """ Compute classifier prediction on each data point in X 

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
        Returns
           out: np.array shape (n, ) - prediction on each data point (number in 0,1,..., num_classes
        """
        out = np.zeros(X.shape[0])
        ### YOUR CODE HERE - 1-4 lines
        A = softmax(np.dot(X,self.W))
        idx = 0
        for row in A:
            out[idx] = np.argmax(row)
            idx += 1
        ### END CODE
        return out
    


def test_encoding():
    print('*'*10, 'Test one-in-k Encoding\n')
    labels = np.array([0, 2, 1, 1])
    m = one_in_k_encoding(labels, 3)
    res =  np.array([[1, 0 , 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    assert res.shape == m.shape, 'encoding shape mismatch'
    assert np.allclose(m, res), m - res
    print('Test Passed')

    
def test_softmax():
    print('*'*5, 'Test softmax\n')
    X = np.zeros((3 ,2))
    X[0, 0] = np.log(4)
    X[1, 1] = np.log(2)
    print('Input to Softmax: \n', X)
    sm = softmax(X)
    expected = np.array([[4.0/5.0, 1.0/5.0], [1.0/3.0, 2.0/3.0], [0.5, 0.5]])
    print('Result of softmax: \n', sm)
    assert np.allclose(expected, sm), 'Expected {0} - got {1}'.format(expected, sm)
    print('Test complete')
        

def test_grad():
    print('*'*5, 'Testing  Gradient\n')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])    
    w = np.ones((2, 3))
    y = np.array([0, 1, 2])
    scl = SoftmaxClassifier(num_classes=3)
    f = lambda z: scl.cost_grad(X, y, W=z)
    numerical_grad_check(f, w)
    print('Test Success')

    
if __name__ == "__main__":
    test_encoding()
    test_softmax()
    test_grad()
