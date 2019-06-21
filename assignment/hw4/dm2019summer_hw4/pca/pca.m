function [eigvector, eigvalue] = pca(data)
%PCA	Principal Component Analysis
%
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%

% YOUR CODE HERE
% calculate cov matrix of data
S = cov(data);
[vectors, value] = eig(S);
% make value vectors
value = sum(value, 2);
% sorting by descend
[value, idx] = sort(value, 'descend');
eigvector = vectors(:, idx);
eigvalue = value;
end