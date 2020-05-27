import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

class kmeans:
    def __init__(self, x1, x2, k):
        self.x1 = x1
        self.x2 = x2
        self.k = k
        self.X = np.array(list(zip(x1, x2)))
        
    # Euclidean distance
    def EuclideanDistance(self, a, b, ax = 1):
        distance = np.linalg.norm(a-b, axis = ax)
        return distance
    
    # return X, cluster labels, coordinates of cluster centers(shape = (15,2))
    def clustering(self):
        
        # initial cluster centers
        np.random.seed(0)
        
        # x coordinates of random cluster center
        C_x = np.random.randint(0, np.max(self.x1)-np.mean(self.x1), size=self.k)
        # y coordinates of random cluster center
        C_y = np.random.randint(0, np.max(self.x2)-np.mean(self.x2), size=self.k)
        self.C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
        
        ## Fill In Your Code Here ##
        k = self.k
        while(1):
            C_b = self.C.copy()
            self.cluster_labels = np.zeros(self.X.shape[0])
            C = np.zeros((k,2))
            labelcnt = np.zeros(k)
            
            for i in range(self.X.shape[0]):
                distance = self.EuclideanDistance(self.C, self.X[i])
                label = np.argmin(distance)
                self.cluster_labels[i] = label
                
                C[label,0] += self.X[i,0]
                C[label,1] += self.X[i,1]
                labelcnt[label] += 1
            
            for j in range(self.k):
                self.C[j,0] = C[j,0] / labelcnt[j]
                self.C[j,1] = C[j,1] / labelcnt[j]
            
            if (self.C == C_b).all():
                break
                
        self.cluster_labels = np.array(self.cluster_labels, dtype=np.int32)
        ############################

        assert self.cluster_labels.shape == (self.X.shape[0],)
        assert self.C.shape == (self.k,2)
        
        return self.X, self.cluster_labels, self.C

       
    def cluster_heterogeneity(self):
        import math
        summ = 0
        ## Fill In Your Code Here ##
        for i in range(len(self.cluster_labels)):
            label = self.cluster_labels[i]
            summ += (self.EuclideanDistance(self.C[label], self.X[i], 0))**2
        heterogeneity = summ
        ############################
        
        return heterogeneity


def plot_data(X, cluster_labels, C, k):
    colors = cm.rainbow(np.linspace(0, 1, k))
    fig = plt.figure(figsize=(10,5))
    
    ## Fill In Your Code Here ##
    defaultcolors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
    x1 = X[:,0]
    x2 = X[:,1]
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.scatter(x1, x2, c = defaultcolors[cluster_labels], s = 1) 
    plt.scatter(C[:,0], C[:,1], c = 'black', marker = '*', s = 700)
    ############################
    
    return plt