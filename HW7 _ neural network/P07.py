import numpy as np
import matplotlib.pyplot as plt



# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)


def ReLU(x):
    ## Fill In Your Code Here ##
    x = np.maximum(0, x)
    ############################
    return x 


def sigmoid(x):
    ## Fill In Your Code Here ##
    x = 1 / (1 + np.exp(-x))
    ############################
    
    return x

def softmax(Xin):
    re = np.zeros(Xin.shape)
    for i in range(Xin.shape[0]):
        re[i] = np.exp(Xin[i])
        re[i] /= np.sum(re[i])
    return re 

# Helper function for forward propagation
def forward_propagation(model, X):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    ## Fill In Your Code Here ##
    h1 = np.dot(X, W1) + b1  # h1 = N * 10
    z1 = ReLU(h1)
    h2 = np.dot(z1, W2) + b2 # h2 = N * 10
    z2 = sigmoid(h2)
    h3 = np.dot(z2, W3) + b3 # h3 = N * 2
    y_hat = softmax(h3)
    ############################
    cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'y_hat': y_hat}
    return y_hat, cache


# Helper function to evaluate the total loss on the dataset
def compute_loss(model, X, y):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    y_hat, _ = forward_propagation(model, X)
    ## Fill In Your Code Here ##
    from math import log
    
    total_loss = 0
    for i in range(X.shape[0]):
        for k in range(2):
            if y[i,k] == 1:
                total_loss -= np.log(y_hat[i,k])
    ############################
    
    return total_loss


# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    y_hat, _ = forward_propagation(model, X)
    ## Fill In Your Code Here ##
    N = y_hat.shape[0]
    prediction = np.zeros((N,1))
    
    for i in range(N):
        if y_hat[i,0] < y_hat[i,1]:
            prediction[i] = 1
    ############################
    
    return prediction


def back_propagation(model, cache, X, y):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    h1, z1, h2, z2, h3, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['y_hat']
    
    ## Fill In Your Code Here ##
    
    N = X.shape[0]
    
    hdim = W2.shape[0]
    
    dW1 = np.zeros(W1.shape)
    db1 = np.zeros(b1.shape)
    dW2 = np.zeros(W2.shape)
    db2 = np.zeros(b2.shape)
    dW3 = np.zeros(W3.shape)
    db3 = np.zeros(b3.shape)
    
    L_h3 = np.zeros(h3.shape)
    L_h2 = np.zeros(h2.shape)
    L_h1 = np.zeros(h1.shape)
    
    y0 = y[:,[0]]
    y1 = y[:,[1]]
    yh0 = y_hat[:,[0]]
    yh1 = y_hat[:,[1]]
    
    L_h3 = np.hstack((-y0*yh1+y1*yh0, -y1*yh0+y0*yh1))
    
    L_b3 = np.sum(L_h3, axis=0) # 1 X 2
    
    h3_z2 = W3 # 10 X 2
    L_W3 = np.dot(L_h3.T, z2).T # (2XN) X (NX10) = 2X10
    L_z2 = np.dot(L_h3, h3_z2.T) # N X 10
    
    sig = z2 * (np.ones(z2.shape) - z2)
    for i in range(N):
        z2_h2 = np.eye(hdim)
        for j in range(hdim):
            z2_h2[j,j] = sig[i,j]
        L_h2[i] = np.dot(L_z2[i], z2_h2) # N X 10
    
    L_b2 = np.sum(L_h2, axis=0) # 1 X 10
    h2_z1 = W2 # 10 X 10
    
    L_W2 = np.dot(L_h2.T, z1).T # (10XN) X (NX10) = 10X10
    L_z1 = np.dot(L_h2, h2_z1.T) # N X 10
    
    for i in range(N):
        z1_h1 = np.zeros((hdim, hdim))
        for j in range(hdim):
            if h1[i,j] > 0:
                z1_h1[j,j] = 1
            elif h1[j,j] == 0:
                z1_h1[j,j] = 0.5            
        L_h1[i] = np.dot(L_z1[i], z1_h1) # N X 10
    
    L_b1 = np.sum(L_h1, axis=0)
    L_W1 = np.dot(L_h1.T, X).T
    
    dW3 = L_W3
    dW2 = L_W2
    dW1 = L_W1
    db3 = L_b3
    db2 = L_b2
    db1 = L_b1
   
        
                        
    ############################
    
    gradients = dict()
    gradients['dW3'] = dW3
    gradients['db3'] = db3
    gradients['dW2'] = dW2
    gradients['db2'] = db2
    gradients['dW1'] = dW1
    gradients['db1'] = db1
    return gradients


def randn_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    W1 = np.random.randn(nn_input_dim, nn_hdim1)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_hdim2)
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.random.randn(nn_hdim2, nn_output_dim)
    b3 = np.zeros((1, nn_output_dim))
    
    return W1, b1, W2, b2, W3, b3


def const_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    # Constant initialization. why problematic? 
    W1 = np.ones((nn_input_dim, nn_hdim1))
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.ones((nn_hdim1, nn_hdim2))
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.ones((nn_hdim2, nn_output_dim))
    b3 = np.zeros((1, nn_output_dim))
 
    return W1, b1, W2, b2, W3, b3


def build_model(X, y, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim,
                lr=0.001, epoch=50000, print_loss=False, init_type='randn'):

    # Initialization
    np.random.seed(0)
    if init_type == 'randn':
        W1, b1, W2, b2, W3, b3 = randn_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim)
    elif init_type == 'const':
        W1, b1, W2, b2, W3, b3 = const_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim)

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    training_loss = []
     
    # Full batch gradient descent. 
    for i in range(epoch):
 
        # Forward propagation
        y_hat, cache = forward_propagation(model, X)
        
        # Backpropagation
        gradients = back_propagation(model, cache, X, y)        

        # Parameter update
        W1 -= lr * gradients['dW1']
        b1 -= lr * gradients['db1']
        W2 -= lr * gradients['dW2']
        b2 -= lr * gradients['db2']
        W3 -= lr * gradients['dW3']
        b3 -= lr * gradients['db3']
         
        # Assign new parameters 
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
        
        # Print the loss.
        if print_loss and (i+1) % 1000 == 0:
            loss = compute_loss(model, X, y)
            print("Loss (iteration %i): %f" %(i+1, loss))
            training_loss.append(loss)
     
    return model, training_loss


