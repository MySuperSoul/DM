vec = [1, 2, 3, 4, 5];
for i=1:length(vec)
    filepath = strcat(num2str(vec(i)), ".gif");
    hack_pca(filepath);
end