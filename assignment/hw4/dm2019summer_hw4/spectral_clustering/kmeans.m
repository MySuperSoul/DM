function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE
vector = randperm(size(X, 1), K);
ctrs = ones(K, size(X, 2));
for i=1:K
    ctrs(i, :) = X(vector(i), :);
end
iter_ctrs = ctrs;
idx = zeros(size(X, 1), 1);

while(true)
    dist_matrix = EuDist2(X, ctrs);
    [Y, idx_k] = min(dist_matrix, [], 2);
    current_idx = idx_k;
    if current_idx == idx
        break;
    end
    idx = current_idx;

    for i=1:K
        ctrs(i, :) = mean(X(idx == i, :), 1);
    end
    iter_ctrs = cat(3, iter_ctrs, ctrs);
end
end
