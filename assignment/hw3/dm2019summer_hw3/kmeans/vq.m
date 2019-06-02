img = imread('sample0.jpg');
fea = double(reshape(img, size(img, 1)*size(img, 2), 3));
% YOUR (TWO LINE) CODE HERE
kset = [8, 16, 32, 64];
for i=1:length(kset)
    figure;
    k = kset(i);
    [idx, ctrs, iter_ctrs] = kmeans(fea, k);
    fea_tmp = ctrs(idx, :);
    fprintf('Finish k -> %d\n', k);
    imshow(uint8(reshape(fea_tmp, size(img))));
    title(sprintf('with k %s', num2str(k)));
    dst_path = sprintf('./Result_images/with_k_%s.jpg', num2str(k));
    imwrite(uint8(reshape(fea_tmp, size(img))), dst_path);
end
