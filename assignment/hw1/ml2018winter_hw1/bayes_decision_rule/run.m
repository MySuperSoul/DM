% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);

%% Part1 likelihood: 
l = likelihood(train_x);

bar(range(1):range(2), l');
xlabel('x');
ylabel('P(x|\omega)');
axis([range(1) - 1, range(2) + 1, 0, 0.5]);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule
num_of_error_test = 0;

for i = 1:size(test_x, 2)
    prob_1 = l(1, i);
    prob_2 = l(2, i);
    if prob_1 > prob_2
        num_of_error_test += test_x(2, i);
    else
        num_of_error_test += test_x(1, i);
    end
end

fprintf("Using maximum likelihood decision rule, Test error is %d\n", num_of_error_test);

%% Part2 posterior:
p = posterior(train_x);

bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
num_of_error_test = 0;
for i = 1:size(test_x, 2)
    prob_1 = p(1, i);
    prob_2 = p(2, i);
    if prob_1 > prob_2
        num_of_error_test += test_x(2, i);
    else
        num_of_error_test += test_x(1, i);
    end
end
fprintf("Using optimal bayes decision rule, Test error is %d\n", num_of_error_test);

%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights

matrix_omega_1 = sum(p .* risk(1, :)');
matrix_omega_2 = sum(p .* risk(2, :)');
matrix_min = min(matrix_omega_1, matrix_omega_2);
risk_min = sum(matrix_min);
fprintf("The minimum risk is %f\n", risk_min);