import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num_iters = 100
    learning_rate = 0.1
    # YOUR CODE HERE
     # begin answer
    X = np.insert(X, 0, values=np.ones((1, N)), axis=0)  # add bias
    for i in range(0, num_iters):
        derive_array = np.sum(np.multiply(X, y - Sigmoid(w, X)), axis=1)
        derive_array = derive_array[:, np.newaxis]
        w = w + learning_rate * derive_array
    return w

def Sigmoid(w, X):
    s = 1.0 + np.exp(-np.dot(w.T, X))
    return 1.0/s