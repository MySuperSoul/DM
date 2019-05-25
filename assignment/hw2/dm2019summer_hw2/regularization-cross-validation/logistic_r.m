function w = logistic_r(X, y, lambda)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
num_iters = 100;
learning_rate = 0.1;
X = [ones(1, size(X, 2));X]; % add bias to X
w = ones(size(X, 1), 1);

for i = 1:num_iters
    w = w - learning_rate * (sum(X .* (Sigmoid(w, X) - y), 2) + 2 * lambda * w);
end
end
