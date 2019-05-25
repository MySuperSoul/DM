import numpy as np
from scipy import optimize

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.ones((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge with any method
    # that support constrain.
    # begin answer
    X = np.insert(X, 0, values=np.ones((1, N)), axis=0)  # add bias
    res = optimize.minimize(func, w, constraints={"fun" : cons, "type" : "ineq", "args" : (X, y)}, method='SLSQP', options={'maxiter': 200})
    w = (res.x)[:, np.newaxis]
    num = np.sum(np.int64(y * np.dot(w.T, X) < 1.01))
    # end answer
    return w, num

def func(w):
    return 0.5 * (np.linalg.norm(w) ** 2)

def cons(w, X, y):
    P, N = X.shape
    w = w.reshape(P, 1)
    s_array = y * np.dot(w.T, X)
    return (np.min(s_array) - 1)

