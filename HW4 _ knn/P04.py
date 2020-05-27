import numpy as np
import collections  # it is optional to use collections

# prediction function is to predict label of one sample using k-NN 
def predict(X_train, y_train, one_sample, k):  
    ## Fill In Your Code Here ##
    ## initializing Dist : sorting first k houses
    from math import sqrt
    
    Dist = []
    Samplenum = []
    for i in range (k):
        diff = one_sample - X_train[i]
        distance = sqrt(np.dot(diff, diff.T))
        Dist.append(distance)
        Samplenum.append(i)
    
    temp = 0
    for i in range (k-1):
        for j in range (i+1, k):
            if Dist[i] >= Dist[j]:
                temp = Dist[j]
                Dist[j] = Dist[i]
                Dist[i] = temp
                temp = Samplenum[j]
                Samplenum[j] = Samplenum[i]
                Samplenum[i] = temp
    
    
    ## Computing rest distances
    for i in range (k, X_train.shape[0]):
        diff = one_sample - X_train[i]
        distance = sqrt(np.dot(diff, diff.T))

        if distance < Dist[0]:
            for j in range (k-1, 0, -1):
                Dist[j] = Dist[j-1]
                Samplenum[j] = Samplenum[j-1]
            Dist[0] = distance
            Samplenum[0] = i
        else:
            for j in range (0, k-1):
                if distance == Dist[j]:
                    for x in range (k-1, j+1, -1):
                        Dist[x] = Dist[x-1]
                        Samplenum[x] = Samplenum[x-1]
                    Dist[j+1] = distance
                    Samplenum[j+1] = i
                    break
               
                if Dist[j] < distance and Dist[j+1] > distance:
                    for x in range (k-1, j+1, -1):
                        Dist[x] = Dist[x-1]
                        Samplenum[x] = Samplenum[x-1]
                    Dist[j+1] = distance
                    Samplenum[j+1] = i
                    break
                    
   
    from collections import Counter
    
    result = []
    for i in range (k):
        result.append(y_train[Samplenum[i]])
    
    cnt = Counter(result)
    pred = cnt.most_common()
    maxi = pred[0][1] 
       
    
    samecnt = []
    for i in pred:
        if i[1] == maxi: 
            samecnt.append(i[0])

        
    pp = 1000
    for i in samecnt:
        if pp > result.index(i):
            pp = result.index(i)
            prediction = result[pp]

    
    ############################
        
    return prediction

# accuracy function is to return average accuracy for test or validation sets 
def accuracy(X_train, y_train, X_test, y_test, k):  # You can use def prediction above.
                                                   
    ## Fill In Your Code Here ##
    N = X_test.shape[0]
    
    correctcnt = 0;
    for i in range (N):
        pred = predict(X_train, y_train, X_test[i], k)
        if pred == y_test[i]:
            correctcnt += 1
    
    acc = correctcnt / N * 100
    ############################         
             
    return acc

# stack_accuracy_over_k is to stack accuracy over k. You can use def accuracy above.         
def stack_accuracy_over_k(X_train, y_train, X_val, y_val):      
    accuracies = []    
    
    ## Fill In Your Code Here ##                                
    for k in range (1, 21):
        acc = accuracy(X_train, y_train, X_val, y_val, k)
        accuracies.append(acc)
    
    ############################
             
    assert len(accuracies) == 20
    return accuracies