X = [0, 0, 0;1, 5, 6;6, 7, 8]
std_matrix = max(X, [], 2) - min(X, [], 2) + eps;
X(X > 0) = 1
