function [vector, idx] = kmin_vector(D, k)
% Input:
% D : n x n matrix
% k : selected k value
% Output:
% vec: 1 x n vector
%     -- first minimum value vector

[Y, idx] = sort(D);
vector = Y(k, :);
idx = idx(k, :);
end