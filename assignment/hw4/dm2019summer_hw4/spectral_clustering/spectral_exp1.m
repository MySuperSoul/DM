load('cluster_data', 'X');

% Choose proper parameters
k_in_knn_graph = 300;
threshold = 0.6;

W = knn_graph(X, k_in_knn_graph, threshold);
idx = spectral(W, 2);
cluster_plot(X, idx);

figure;
idx = litekmeans(X, 2);
cluster_plot(X, idx);
