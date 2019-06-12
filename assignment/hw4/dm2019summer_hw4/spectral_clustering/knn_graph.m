function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE
distance_matrix = EuDist2(X);
sprintf("Finish distance calculate");
vec = kmin_vector(distance_matrix, k + 1);
sprintf("Finish kmin vector calculate");
W = distance_matrix <= vec;
W = W - eye(size(W, 1));
% filter threshold
W = W & (distance_matrix <= threshold);
W = W | W'; % make symmetric matrix

sprintf("Finish knn_graph");
end
