import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer
    converge_flag = False
    while True:
        if converge_flag: break
        converge_flag = True # initial to true
        iters += 1
        for i in range(0, N):
            x = np.insert(X[:, i], 0, [1], 0)
            s = np.dot(np.transpose(w), x) * y[:, i]
            if (s <= 0):
                matrix = x * y[:, i]
                w = w + matrix.reshape(matrix.shape[0], 1)
                converge_flag = False
    # end answer
    
    return w, iters