## Assignment2
---

### 1. A Walk Through Linear Models

**(a) Perceptron**

- **(i)** 

&nbsp;&nbsp;&nbsp;&nbsp;Since the data generated in `mkdata.py` is linear seperable, so that the perceptron algoorithm will converge finally, that's to say the train error is **0** theoretical. 
&nbsp;&nbsp;&nbsp;&nbsp;To generate test data and train data, I fit `Ttrain + Ttest` into `mkdata`, then through slice for `X and y` to get `X_train X_test y_train y_test`.
&nbsp;&nbsp;&nbsp;&nbsp;The following is my results :

    -> When the number of training set is 10, the training error rate 
       is 0, and the testing error rate is 0.11166
    -> When the number of training set is 100, the training error rate
       is 0, and the testing error rate is 0.01448

<br>

- **(ii)**

    - When the number of training set is `10`, the average number of iterations before converge is `5.353`
    - When the number of training set is `100`, the average number of iterations before converge is `38.921`
<br>

- **(iii)**
&nbsp;&nbsp;&nbsp;&nbsp;As for the training data is not linear seperable situation, the most significant effect is that **the iteration loop is endless**, because the perceptron algorithm is trying to find parameters that can seperate the training data exactly. And in my program, I test it and find exactly that this program cell part is running endless.

<br>

**(b) Linear Regression**

- **(i)**
&nbsp;&nbsp;&nbsp;&nbsp;In this section, I also fit the `Ttrain + Ttest` into `mkdata` to get my training set and test set. The training error rate is `0.03766 or 3.766%`, the test error rate is `0.04735 or 4.735%`. And the linear-regression parameter is get from $\hat{\omega} = (XX^T)^{-1}Xy$, the implementing code is below:
```python
    X = np.insert(X, 0, values=np.ones((1, N)), axis=0) # add bias
    s = np.linalg.pinv(np.dot(X, X.transpose())) # (X X.T)^-1
    w = np.dot(np.dot(s, X), y.T) # * X * y
```

<br>

- **(ii)**
&nbsp;&nbsp;&nbsp;&nbsp;As for the situation that the training data is noisy, so the test data lable I use $y_{test} = \omega_f^T \cdot x_{test}$ to generate the test data lable, So that the test data has no noisy while the training data set remain some noisy. Then set `Ttrain = Ttest = 100`, I get that the `training error rate ` is **0.1309 or 13.09%**, the `testing error rate ` is **0.05809 or 5.809%**.

<br>

- **(iii)**
&nbsp;&nbsp;&nbsp;&nbsp;In this problem, first add bias term to both training set and testing set of X. Then to calculate `num_of_error` using $\omega^TXy < 0$ rule. Finally divide the `Ttrain(X.shape[1])` and `Ttest(X_test.shape[1])` to get final error rate of training set and testing set.
&nbsp;&nbsp;&nbsp;&nbsp;The result of `training error rate` is **0.49 or 49%**.
&nbsp;&nbsp;&nbsp;&nbsp;The result of `testing error rate` is **0.5496 or 54.96%**.

<br>

- **(iv)**
&nbsp;&nbsp;&nbsp;&nbsp;In this transformation problem, I first do the transform $(x_1, x_2) \to (x_1, x_2, x_1x_2, x_1^2, x_2^2)$, and then add the bias term to become like this $(1, x_1, x_2, x_1x_2, x_1^2, x_2^2)$, and in this way, I calculate out that the `training error rate` is **5% or 0.05**. The `testing error rate` is **6.6% or 0.066**. The implementing of transformation is like this:
```python
    X_t = np.row_stack((X, X[0, :] * X[1, :]))
    X_t = np.row_stack((X_t, np.power(X[0, :], 2)))
    X_t = np.row_stack((X_t, np.power(X[1, :], 2)))
    add_bias_xtrain = np.insert(X_t, 0, 
                            values=np.ones((1, X_t.shape[1])), 
                            axis=0)
```