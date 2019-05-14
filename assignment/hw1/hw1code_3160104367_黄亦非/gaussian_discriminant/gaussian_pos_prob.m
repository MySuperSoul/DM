function P = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);
K = length(Phi);
P = zeros(N, K);

% Your code HERE
for i=1:K
    phi_i = Phi(i);
    Sigma_i = Sigma(:, :, i);
    mu_i = Mu(:, i);
    deteminant = det(Sigma_i);
    Sigma_i_inv = pinv(Sigma_i);
    post_i = phi_i / (2 * pi * sqrt(deteminant)) * ...
      exp(-0.5 * sum((X - mu_i)' * Sigma_i_inv .* (X - mu_i)', 2));
    P(:, i) = post_i;
end
evidence_matrix = sum(P, 2);
P = P ./ evidence_matrix;