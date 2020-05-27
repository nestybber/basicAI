import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# P1
def read_csv_using_pandas(csv_path='exam_scores.csv'):

    ## Fill In Your Code Here ##
    data = pd.read_csv(csv_path)

    ############################

    print(data.shape)
    print(data.head())
    

    return data


def parse_pd_data(data, fields=['Circuit',
                                'DataStructure',
                                'MachineIntelligence']):
    values = []
    
    for i in fields:
        values.append(data[i])
    
    values = np.array(values)
    return values


def plot_data(values):
    assert len(values) == 3
    assert type(values[0]) == np.ndarray
    figsize = (6, 4)
    title_fontsize = 20
    label_fontsize = 15

    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    
    ## Fill In Your Code Here ##
    # scatter
    ax.scatter(values[0],values[1],values[2])
    # set title. use title_fontsize above.
    ax.set_title('Score Distributions', fontsize = title_fontsize)
    # set labels for each axes. use label_fontsize above.
    ax.set_xlabel('Circuit', fontsize = label_fontsize)
    ax.set_ylabel('DataStructure', fontsize = label_fontsize)
    ax.set_zlabel('MachineIntelligence', fontsize = label_fontsize)
    ############################
    plt.show()
    return fig


# P2
def prepare_dataset_for_linear_regression(values):

    bias = np.ones(len(values[0]))
    X = np.array([bias, values[0], values[1]]).T # X = N * 3
    y = np.array(values[2])                      # y = N * 1

    return X, y


class LinearRegression:

    def __init__(self, lr=0.0001, iterations=100000):
        self.lr = lr
        self.iterations = iterations
        self.average_rss_history = []

    def fit(self, X, y):
        N = len(y)

        ## Fill In Your Code Here ##
        # initialize w
        self.w = np.zeros((3,1))   # w = 3 * 1
        y = y.reshape(1000,1)    # y = N * 1

        ############################

        for i in range(self.iterations):

            ## Fill In Your Code Here ##
            # implement gradient descent
            Guess = np.dot(X, self.w)    # Guess = N * 1
            self.w = self.w - self.lr * 2 * np.dot(X.T, (Guess - y)) / N
            average_rss = np.dot((y - Guess).T,(y - Guess)) / N
            average_rss = average_rss[0]

            ############################
            self.average_rss_history.append(average_rss)


    def predict(self, X):
        ## Fill In Your Code Here ##
        pred_y = np.dot(X, self.w)
        ############################
        return pred_y


def plot_average_rss_history(iterations, history):
    figsize = (6,4)
    title_fontsize = 20
    label_fontsize = 15

    # plot rss_avg history over iterations
    fig = plt.figure(figsize=figsize)
    plt.ylim(0,100)

    ## Fill In Your Code Here ##
    xx = np.linspace(0, iterations, iterations)   ## fixing iteration number(error)

    # plot
    # set title
    # set labels for axes
    plt.plot(xx, history)
    plt.title('Average RSS over number of iterations', fontsize = title_fontsize)
    plt.xlabel('iterations', fontsize = label_fontsize)
    plt.ylabel('Average RSS', fontsize = label_fontsize)

    ############################

    return fig


# P3
def plot_data_with_wireframe(values, w, wireframe_color='red'):
    assert len(w) == 3
    title_fontsize = 20
    label_fontsize = 15
    figsize = (6,4)

    def make_meshgrids(x, y, num=10):

        ## Fill In Your Code Here ##
        # make meshgrids for 3D plot.
        # HINT : use np.linspace function
        x_linspace = np.linspace(min(x), max(x), num)
        y_linspace = np.linspace(min(y), max(y), num)

        ############################

        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
        return x_grid, y_grid

    x_grid, y_grid = make_meshgrids(values[0], values[1])

    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)

    ## Fill In Your Code Here ##
    result = w[0]+w[1]*x_grid+w[2]*y_grid       # Ploting the Plane w0 + w1*value[0] + w2*value[1]
    
    # scatter
    ax.scatter(values[0],values[1],values[2])
    # plot wireframe
    ax.plot_wireframe(x_grid, y_grid, result, color = wireframe_color, linewidth = 1)

    # set title
    ax.set_title('Score Distributions', fontsize = title_fontsize)

    # set labels for axes
    ax.set_xlabel('Circuit', fontsize = label_fontsize)
    ax.set_ylabel('DataStructure', fontsize = label_fontsize)
    ax.set_zlabel('MachineIntelligence', fontsize = label_fontsize)
    ############################

    plt.show()
    return fig


def get_closed_form_solution(X, y):

    w = np.zeros(X.shape[1])
    
    ## Fill In Your Code Here ##
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    w = w.reshape(3,1)
    ############################

    return w


################################################ Additional Class

class ranparamLR(LinearRegression):
    def fit(self, X, y):
        N = len(y)

        ## Fill In Your Code Here ##
        # initialize w
        self.w = np.random.random((3,1))   # w = 3 * 1
        y = y.reshape(1000,1)    # y = N * 1

        ############################

        for i in range(self.iterations):

            Guess = np.dot(X, self.w)    # Guess = N * 1
            self.w = self.w - self.lr * 2 * np.dot(X.T, (Guess - y)) / N
            average_rss = np.dot((y - Guess).T,(y - Guess)) / N
            average_rss = average_rss[0]

            ############################
            self.average_rss_history.append(average_rss)

    