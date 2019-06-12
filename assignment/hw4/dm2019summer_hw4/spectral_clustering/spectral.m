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
D(logical(eye(size(D)))) = sum_w_matrix;
[y_matrix, value_matrix] = eig(D - W, D);
y = y_matrix(:, 2); % second smallest eigen vector
sprintf("Finish eigen y calculate");
idx = litekmeans(y, k);
sprintf("Finish spectral");
end
