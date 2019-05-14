%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier
x_likelihood = likelihood(x);
c_spam_pro = x_likelihood(2, :) ./ x_likelihood(1, :);
[b, i] = sort(c_spam_pro, 'descend');

log_likelihood = log(x_likelihood);
log_p_spam = log(num_spam_train / (num_ham_train + num_spam_train));
log_p_ham = log(num_ham_train / (num_ham_train + num_spam_train));

% test ham classifier
ham_test_classifier = ham_test * log_likelihood(1, :)' + log_p_ham;
spam_test_classifier = ham_test * log_likelihood(2, :)' + log_p_spam;
bayesian_classfier = ham_test_classifier > spam_test_classifier;

error_num_on_ham_test = size(find(bayesian_classfier == 0), 1);
fp = error_num_on_ham_test;
tn = size(bayesian_classfier, 1) - fp;

% test spam classifier
ham_test_classifier = spam_test * log_likelihood(1, :)' + log_p_ham;
spam_test_classifier = spam_test * log_likelihood(2, :)' + log_p_spam;
bayesian_classfier = spam_test_classifier > ham_test_classifier;

error_num_on_spam_test = size(find(bayesian_classfier == 0), 1);
fn = error_num_on_spam_test;
tp = size(bayesian_classfier, 1) - fn;

error_rate = (error_num_on_ham_test + error_num_on_spam_test) / (size(ham_test, 1) + size(spam_test, 1));
correct_rate = (1 - error_rate) * 100;
fprintf("The correct rate on test set is %f%%\n", correct_rate);

precision = tp / (tp + fp);
recall = tp / (tp + fn);
fprintf("fp: %d --- fn: %d --- tp: %d --- tn: %d\n", fp, fn, tp, tn);
fprintf("The precision is %f\n The recall is %f\n", precision, recall);