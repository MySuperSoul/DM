load('kmeans_data', 'X');
[idx, ctrs, iter_ctrs] = kmeans(X, 2);

max_dist = -9e10;
min_dist = -max_dist;

for i=1:1000
    [idx, ctrs, iter_ctrs] = kmeans(X, 2);
    dist = 0;
    for j=1:size(ctrs, 1)
        dist += sum(sumsq(X(idx == j, :) - ctrs(j, :), 2));
    end
    
    if(dist > max_dist)
        max_idx = idx; max_ctrs = ctrs; max_iter_ctrs = iter_ctrs; max_dist = dist;
    end
    if(dist < min_dist)
        min_idx = idx; min_ctrs = ctrs; min_iter_ctrs = iter_ctrs; min_dist = dist;
    end
end

kmeans_plot(X, max_idx, max_ctrs, max_iter_ctrs);
title('max SD');
kmeans_plot(X, min_idx, min_ctrs, min_iter_ctrs);
title('min SD');