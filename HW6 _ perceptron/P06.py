import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_samples(X_train, y_train):
    fig = plt.figure(figsize=(16,7))
    
    ## Fill In Your Code Here ##
    samplenum = 0;
    n = 28;
    for samplenum in range(30):
        pic = fig.add_subplot(3,10,samplenum+1)
        plt.imshow(X_train[samplenum, :784].reshape(n,n))
        plt.title('Number %d' % (y_train[samplenum]*(-1)+4))
    ############################    
    
    return fig

def sgn(x):
    return (x >= 0)*2-1

# it is to predict accuracy of X data.
def predict(X, y, w):
    
    ## Fill In Your Code Here ##
    correct = 0
    for i in range(X.shape[0]):
        pred = sgn(np.dot(X[i,:], w))
        if pred == y[i]:
            correct += 1
            
    accuracy = correct / X.shape[0]
    ############################
    
    return accuracy

# return w, number_of_misclassifications, test_accuracy
def perceptron(X, y, w, epoch):
   
    number_of_misclassifications = []
    test_accuracy = []
    
    ## Fill In Your Code Here ##
    epochcnt = 0
    for epochcnt in range(epoch):
        miss = 0
        for i in range(X['train'].shape[0]):
            pred = sgn(np.dot(X['train'][i,:], w))
            if pred == y['train'][i]:
                w = w
            else:
                w = w + y['train'][i]*X['train'][i,:]
                miss += 1
        number_of_misclassifications.append(miss)
        test_accuracy.append(predict(X['test'], y['test'], w))
    ############################
    
    return w, number_of_misclassifications, test_accuracy

# plot number_of_misclassifications returned by perceptron
def plot_number_of_misclassifications_over_epochs(errors):
    
    fig = plt.figure(figsize=(17,5))
    
    ## Fill In Your Code Here ##
    plt.plot(np.arange(1,101,step=1), errors)
    plt.title('Number of misclassifications over epochs', fontsize=25)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Number of misclassifications', fontsize=20)
    ############################
    
    return fig

# plot test_accuracy returned by perceptron
def plot_accuracy_over_epochs(test_accuracy):
    
    fig = plt.figure(figsize=(17,5))
    
    ## Fill In Your Code Here ##
    plt.plot(np.arange(1,101,step=1), test_accuracy)
    plt.title('Accuracy over epochs', fontsize=25)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    ############################
    
    return fig