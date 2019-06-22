function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE
distance_matrix = EuDist2(X);
fprintf("Finish distance calculate\n");
[vector, idx] = kmin_vector(distance_matrix, k);
fprintf("Finish kmin vector calculate\n");

W = zeros(size(distance_matrix));
% consider threshold
W = distance_matrix < min([vector;threshold * ones(size(vector))], [], 1);
W = W | W'; % make symmetric matrix

fprintf("Finish knn_graph\n");
end