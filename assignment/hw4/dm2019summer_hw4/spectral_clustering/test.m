A = [0, 2, 3;2, 0, 6;3, 6, 0]
B = [1, 2, 3;7, 8, 9;6, 5, 4];
% spectral(A, 2);
w = knn_graph(A, 1, 3);
load('cluster_data', 'X');