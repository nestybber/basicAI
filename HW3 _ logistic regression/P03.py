import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(df):
    """Return X, y and features.
    
    Args:
        df: pandas.DataFrame object.
    
    Returns:
        Tuple of (X, y, features).
        X (ndarray): include the columns of the features and bias, shape == (N, D+1)
        y (ndarray): label vector, shape == (N, )
        feature (ndarray): names for each features except bias, shape == (D, )
    """
    
    N = df.shape[0] # the number of samples
    D = df.shape[1] - 1 # the number of features, excluding a label
    
    ## Fill In Your Code Here ##
    df.insert(D+1, 'intercept', 1)
    features = df.columns
    y = np.array(df[features[0]])
    X = np.array([df[features[i]] for i in range (1, D+2)])
    X = X.T
    features = features[1:D+1]
    ############################
    assert X.shape == (N, D+1) and y.shape == (N, )
    
    return (X, y, features)


def sigmoid(x):
    """Sigmoid function""" # return value = sigmoid(x)
    ## Fill In Your Code Here ##
    ret = 1 / (1 + np.exp(-x))
    ############################
    return ret


class LogisticRegressionSGD():
    
    def __init__(self, lr=0.8, iterations=100000, number_of_ensemble=1000):
        self.lr = lr
        self.iterations = iterations
        self.number_of_ensemble = number_of_ensemble
    
    
    def initialize_w(self, D):
        """Initialize w to ndarray zeros vector with shape == (D, )"""
        
        ## Fill In Your Code Here ##
        w = np.zeros((D,))
        ############################
        assert w.shape == (D, )
        return w
        
        
    def predict(self, X, w):
        """Predict labels(0 or 1) for X from inffered parameters w."""
        ## Fill In Your Code Here ##
        score = np.dot(X, w)
        pred = sigmoid(score).round()
         
                      
        ############################
        return pred
    
    
    def get_train_sample(self, X, y, i):
        """Perform a training data sampling.
        
        Since we assumed that the train set had been shuffled randomly, 
        it's fine to sample the data sequentially.
        """
        N = X.shape[0]
        D = X.shape[1]
        
        ## Fill In Your Code Here ##
        i = i % N
        X_ = X[i]
        y_ = y[i]
        X_ = X_.reshape(1, D)
        y_ = y_.reshape(1,)
        ############################
        
        assert X_.shape == (1, D) and y_.shape == (1,)
        return X_, y_
        
    def get_loss(self, X, y, w):
        """Calculate classification error using def predict above"""
        ## Fill In Your Code Here ##
        pred = self.predict(X, w)
        error = np.dot((pred - y).T, (pred - y)) / X.shape[0]
        ############################                      
        return error
    
    
    def fit(self, X, y):
        """Perform logistic regression using SGD.
        
        There are three instance variables to solve P1.2, 1.3, 2.1, 2.2
        1. avg_loss_over_itr
        2. test_error_over_iter
        3. w_ensemble
                        
        """
        N = X['train'].shape[0]
        D = X['train'].shape[1]
        
        self.avg_loss_over_itr = []
        self.test_error_over_itr = []
        self.w_ensemble = self.initialize_w(D)
        
        w = self.initialize_w(D)
        tot_loss = 0
        
        for i in range(self.iterations):
            
            X_, y_ = self.get_train_sample(X['train'], y['train'], i)
            tot_loss +=  self.get_loss(X_, y_, w)
            
            ## Fill In Your Code Here ##
            # gradient descent
            Degree = sigmoid(np.dot(X_, w))
            for j in range(D):
                w[j] = w[j] - self.lr * X_[0][j] * (Degree - y_[0])
            ############################
            
            if (i + 1) % 100 == 0:
                avg_loss = tot_loss / (i + 1)
                test_error = self.get_loss(X['test'], y['test'], w)
                self.avg_loss_over_itr.append(avg_loss)
                self.test_error_over_itr.append(test_error)

            if self.iterations - (i+1) < self.number_of_ensemble:
                self.w_ensemble += w / self.number_of_ensemble

        return w
    
    
    def get_accuracy(self, X, y, w):
        """Return accuracy for given X, y and w in percentage.
        
        Accuracy is the fraction of predictions our model got right.
        """
        ## Fill In Your Code Here ##
        N = X.shape[0]
        
        result = self.predict(X, w)
        
        cnt = 0
        for i in range(N):
            if y[i] == result[i]:
                cnt += 1
                
        accuracy = cnt / N
        ############################
        return accuracy
        
        
def get_indices_of_fields(fields, features):
    """return a list contains indices of fields. """
    ## Fill In Your Code Here ##
    indices = []
    
    for i in range(len(fields)):
        for j in range(len(features)):
            if fields[i] == features[j]:
                indices.append(j)
                
    ############################
    assert len(indices) == len(fields)
    return indices
