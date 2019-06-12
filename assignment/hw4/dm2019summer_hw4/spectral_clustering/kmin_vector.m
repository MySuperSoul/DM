function vec = kmin_vector(D, k)
% Input:
% D : n x n matrix
% k : selected k value
% Output:
% vec: 1 x n vector
%     -- k minimum value vector

[Y, idx] = sort(D);
vec = Y(k, :);
end