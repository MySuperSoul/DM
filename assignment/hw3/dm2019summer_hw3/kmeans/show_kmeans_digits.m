kset = [10, 20, 50];
load('digit_data', 'X');
for i=1:length(kset)
    [idx, ctrs, iter_ctrs] = kmeans(X, kset(i));
    figure;
    show_digit(ctrs);
    title(strcat('k = ', num2str(kset(i))));
end