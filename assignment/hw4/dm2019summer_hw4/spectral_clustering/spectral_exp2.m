load('TDT2_data', 'fea', 'gnd');
warning('off');
% YOUR CODE HERE
k = numel(unique(gnd));
accuracy_sp = 0;
accuracy_kmeans = 0;
mihat_sp = 0;
mihat_kmeans = 0;
times = 20;

for i=1:times
    res = litekmeans(fea, k, 'Replicates', 10);
    res = bestMap(gnd, res);
    accuracy_kmeans = accuracy_kmeans + length(find(gnd == res))/length(gnd);
    mihat_kmeans = mihat_kmeans + MutualInfo(gnd,res);
    fprintf("Finish kmeans\n");

    options = [];
    options.NeighborMode = 'KNN';
    options.k = 7;
    options.WeightMode = 'Binary';
    W = constructW(fea,options);
    fprintf("Finish W construct\n");

    res = spectral(W, 3);
    res = bestMap(gnd, res);
    accuracy_sp = accuracy_sp + length(find(gnd == res))/length(gnd);
    mihat_sp = mihat_sp + MutualInfo(gnd, res);
    fprintf("Finish -> %d\n\n", i);
end

fprintf("Accuracy for spectral is %f, mutual information is %f\n", accuracy_sp / times, mihat_sp / times);
fprintf("Accuracy for kmeans is %f, mutual information is %f\n", accuracy_kmeans / times, mihat_kmeans / times);