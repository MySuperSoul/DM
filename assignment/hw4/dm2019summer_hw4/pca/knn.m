function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%

% YOUR CODE HERE
dist_matrix = EuDist2(X', X_train');
[Y, idx] = sort(dist_matrix, 2);
idx_k = idx(:, 1:K);
if(K == 1)
    y = y_train(idx_k);
else
    y = mode(y_train(idx_k), 2)';
end
end

