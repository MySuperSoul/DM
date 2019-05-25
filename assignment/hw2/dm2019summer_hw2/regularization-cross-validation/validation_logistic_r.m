%% Ridge Regression
load('digit_train', 'X', 'y');
% Do feature normalization
std_matrix = std(X, 0, 2) + eps; % add eps for 0 situation
X = (X - mean(X, 2)) ./ std_matrix;

% ...
%show_digit(X)
% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
E_val_store = zeros(size(lambdas, 2), 1);
w_store = zeros(size(X, 1) + 1, length(lambdas));
w_best = zeros(size(X, 1) + 1, 1);
E_val_min = 10000;

adjust_y_train = y;
% change label to {0, 1} when training.
adjust_y_train(adjust_y_train < 0) = 0;

for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        X_ = X(:, [1 : j - 1, j + 1 : size(X, 2)]);
        y_ = adjust_y_train(:, [1 : j - 1, j + 1 : end]); % take point j out of X
        w = logistic_r(X_, y_, lambdas(i));
        E_val = E_val + (w' * [ones(1, 1);X(:, j)] * y(j) < 0); % testing using {-1, 1} label
        fprintf("For lambda %d -> Finish step %d Total:200\n", lambdas(i), j);
    end
    E_val_store(i) = E_val;
    w_store(:, i) = w;
    % Update lambda according validation error

    if (E_val < E_val_min)
        E_val_min = E_val;
        lambda = lambdas(i);
        w_best = w;
    end
    fprintf("Error number is %d\n", E_val);
end

fprintf("\n lambda choosen by LOOCV is %f\n", lambda);
% Compute training error
E_train = sum((w_best' * [ones(1, size(X, 2));X] .* y) < 0) / size(X, 2);
load('digit_test', 'X_test', 'y_test');
std_matrix = std(X_test, 0, 2) + eps;
X_test = (X_test - mean(X_test, 2)) ./ std_matrix;
E_test = sum((w_best' * [ones(1, size(X_test, 2));X_test] .* y_test) < 0) / size(X_test, 2);
fprintf("With Regularization lambda %f: training error: %f, testing error: %f\n\n", lambda, E_train, E_test);


zero_lambda_index = find(lambdas == 0);
w_zero = w_store(:, zero_lambda_index);
E_train = sum((w_zero' * [ones(1, size(X, 2));X] .* y) < 0) / size(X, 2);
E_test = sum((w_zero' * [ones(1, size(X_test, 2));X_test] .* y_test) < 0) / size(X_test, 2);
fprintf("Without Regularization: training error: %f, testing error: %f\n\n", E_train, E_test);