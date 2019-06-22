function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE
sum_w_matrix = sum(W, 2);
D = zeros(size(W));

for i=1:size(D, 1)
    D(i, i) = sum_w_matrix(i);
end

fprintf("Finish D calculate\n");
[y_matrix, value_matrix] = eig(D - W, D);
y = y_matrix(:, 2); % second smallest eigen vector
fprintf("Finish eigen y calculate\n");
idx = litekmeans(y, k); % perform kmeans
fprintf("Finish spectral\n");
end
