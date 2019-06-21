% YOUR CODE HERE

load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');

% YOUR CODE HERE

% 1. Feature preprocessing
[num_train, num_featrues] = size(fea_Train);
[num_test, num_featrues] = size(fea_Test);
% 2. Run LDA
options = [];
options.Fisherface = 1;

[eigenvectors, eigenvalues] = LDA(gnd_Train', options, fea_Train);
% 4. Project data on to low dimensional space
reduced_dim = [8, 16, 32, 39]; % only 39 dim
% 5. Run KNN in low dimensional space
for i=1:length(reduced_dim)
    current_dim = reduced_dim(i);
    transform_matrix = eigenvectors(:, 1:current_dim);
    low_dim_train = fea_Train * transform_matrix;
    low_dim_test = fea_Test * transform_matrix;

    predict_test = knn(low_dim_test', low_dim_train', gnd_Train', 1);
    predict_test = predict_test';
    correct_num = sum(predict_test == gnd_Test);

    fprintf("For dim %d, the testing error rate is %f\n", current_dim, 1 - (correct_num / num_test));
end