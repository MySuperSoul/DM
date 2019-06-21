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
vec = kmin_vector(distance_matrix, k);
fprintf("Finish kmin vector calculate\n");
W = distance_matrix <= vec;
% filter threshold
W = W & (distance_matrix <= threshold);
W = W | W'; % make symmetric matrix

fprintf("Finish knn_graph\n");
end
