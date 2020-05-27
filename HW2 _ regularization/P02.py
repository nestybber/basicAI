import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def insert_intercept(dataframe):
    dataframe.insert(1, 'intercept', 1)
    return dataframe

def split_data(dataframe):

    fields = dataframe.columns
    
    y = np.array(dataframe[fields[0]])
    X = np.array([dataframe[fields[i]] for i in range (1, len(fields))])
   
    

    ############################
    assert type(X) == np.ndarray
    assert type(y) == np.ndarray

    return X, y

def CoordinateLasso(X, y, lambda_):
    np.random.seed(0)
    training_error_history = []
    
    ## Fill In Your Code Here ##
    # initialize w
    D = len(X)
    N = len(y)
    
    w = np.random.normal(0,1,D)
    w_before = np.random.normal(0,1,D)
    excptjth = np.random.normal(0,1,N)
    test = 1
    
    # X = D * N
    # y = N * 1
    # w = D * 1
    y = y.reshape(N,1)
    w = w.reshape(D,1)
    w_before = w_before.reshape(D,1)
    excptjth = excptjth.reshape(N,1)
    
    Normalizer = np.sum(np.square(X), axis=1)
    Normalizer = Normalizer.reshape(D,1) # Normalzer = D * 1
    
    while test > 0.000001:
        w_before = w.copy()
        
        # Calculating for next w
        for j in range(0, D):
            wecpj = w.copy()
            rhoj = 0
            wecpj[j] = 0
            Guess = np.dot(X.T, wecpj)            
            Rssj = y - Guess
            rhoj = np.dot(X[j,:], Rssj)[0]
            if j == 0:
                w[j] = rhoj / Normalizer[j]          
            elif rhoj < (-1)*lambda_/2:
                w[j] = (rhoj + lambda_/2) / Normalizer[j]
            elif rhoj > lambda_/2:
                w[j] = (rhoj - lambda_/2) / Normalizer[j]
            else:
                w[j] = 0
                 
        # My prediction           
        RSS = np.dot((y-Guess).T, (y-Guess))
        
        training_error_history.append(RSS[0][0])
        test = max(np.abs(w - w_before))

    ############################
    
            
    return w, training_error_history

def plot_error_over_iterations(error_history):
    
    fig = plt.figure(figsize =(30,7))
    
    ## Fill In Your Code Here ##
    iterations = len(error_history)
    it = np.linspace(0, iterations, iterations)
    
    plt.plot(it, error_history, marker = '*', markersize = 20)
    plt.title('Training error over iterations', fontsize = 30)
    plt.xlabel('iterations', fontsize = 30)
    plt.ylabel('Squared errors', fontsize = 30)
    
    ############################
    
    return fig
        
def stack_weights_by_lambda(lambda_ , X, y):
    
    ## Fill In Your Code Here ##
    D = len(X)
    w, error = CoordinateLasso(X, y, lambda_[0])
    w_tot = w.reshape(1,D)[0]
    
    for j in range(1, 10):
        w, error = CoordinateLasso(X, y, lambda_[j])
        w = w.reshape(1,D)[0]
        w_tot = np.vstack([w_tot,w])  

    ############################    
    assert w_tot.shape == (10,96)
    
    return w_tot
    
    
def plot_weights(lambda_, w_tot, dataframe, features):
    
    fig = plt.figure(figsize=(30,7))
    
    ## Fill In Your Code Here ##
    
    D = len(dataframe.columns)   # D = 96
    N = len(lambda_)             # N = 10
    wsel = np.zeros((N,len(features)))

        
    from math import log
    xx = np.zeros(10)
    for i in range(N):
        xx[i] = log(lambda_[i])
    
    featnum = 0
    for feat in features: 
        for j in range(D):
            if dataframe.columns[j] == feat:
                for i in range(N):
                    wsel[i][featnum] = w_tot[i][j-1]
                featnum += 1
                
    wsel = wsel.T
    
    for i in range(len(features)):
        plt.plot(xx, wsel[i,:], marker = '*', markersize = 20)
        
    
    plt.title('Regularization paths', fontsize = 30)
    plt.xlabel('log(λ)', fontsize = 30)
    plt.ylabel('Weights', fontsize = 30)
    plt.legend(features, fontsize = 20)
    plt.xlim(max(xx), 0)
    ############################ 

    return fig
      
    
def plot_training_error(lambda_, w_tot,  X, y):
    
    fig = plt.figure(figsize=(30,7))
    
    ## Fill In Your Code Here ##
    N = len(y)   # N
    K = len(lambda_)  # K = 10
    
    y = y.reshape(1,N)
    
    from math import log
    
    xx = np.zeros(10)
    RSS = np.zeros(10)
    
    for i in range(K):
        xx[i] = log(lambda_[i]) 
    
    Guess = np.dot(w_tot,X)
    
    for i in range(K):
        RSS[i] = np.dot((y - Guess[i,:].reshape(1,N)[0]),(y - Guess[i,:].reshape(1,N)[0]).T)
        
    plt.plot(xx, RSS, marker = '*', markersize = 20)
    plt.title('Training errors over log(λ)', fontsize = 30)
    plt.xlabel('log(λ)', fontsize = 30)
    plt.ylabel('Squared errors', fontsize = 30)        
    ############################ 

    return fig


def plot_test_error(lambda_, w_tot,  X, y):
    
    fig = plt.figure(figsize=(30,7))
    
    ## Fill In Your Code Here ##
    N = len(y)   # N
    K = len(lambda_)  # K = 10
    
    y = y.reshape(1,N)
    
    from math import log
    
    xx = np.zeros(10)
    RSS = np.zeros(10)
    
    for i in range(K):
        xx[i] = log(lambda_[i]) 
    
    Guess = np.dot(w_tot,X)
    
    for i in range(K):
        RSS[i] = np.dot((y - Guess[i,:].reshape(1,N)[0]),(y - Guess[i,:].reshape(1,N)[0]).T)

    plt.plot(xx, RSS, marker = '*', markersize = 20)
    plt.title('Test errors over log(λ)', fontsize = 30)
    plt.xlabel('log(λ)', fontsize = 30)
    plt.ylabel('Squared errors', fontsize = 30)    
    ############################ 

    return fig

def plot_number_of_nonzero_index(lambda_, w_tot):
    
    fig = plt.figure(figsize=(30,7))
    
    ## Fill In Your Code Here ##
    D = len(w_tot.T)
    K = len(lambda_)
    
    numofzero = np.zeros((1,K))[0]
    for i in range(K):
        for j in range(D):
            if w_tot[i][j] == 0:
                numofzero[i] += 1
                
    numofnonzero = D * np.ones((1,K))[0] - numofzero
    
    plt.plot(lambda_, numofnonzero, marker = '*', markersize = 20)
    plt.title('Number of non-zero weights', fontsize = 30)
    plt.xlabel('λ', fontsize = 30)
    plt.ylabel('Number of non-zero weights', fontsize = 30)        
    ############################ 

    return fig