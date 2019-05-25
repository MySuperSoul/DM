function w = ridge(X, y, lambda)
%RIDGE Ridge Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
%
% NOTE: You can use pinv() if the matrix is singular.

% YOUR CODE HERE
P = size(X, 1);
N = size(X, 2);

w = ones(P + 1, 1);
X = [ones(1, N);X]; % add bias

w = pinv(X * X' + lambda * eye(P+1)) * X * y'; 
end
