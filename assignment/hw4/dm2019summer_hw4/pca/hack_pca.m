function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation
img_o = imread(filename);
img_r = double(img_o);

% YOUR CODE HERE
% here use position in image to represent features
% filter non background piexls
[x, y] = find(img_r < 255);
data = [x, y];
[eigvectors, eigvalues] = pca(data);

eigenvector = eigvectors(:, 1)';
rotate_angle = rad2deg(atan2(eigenvector(1), eigenvector(2)));
new_image_r = imrotate(img_o, rotate_angle);
new_image_r(find(double(new_image_r) < 1)) = 255;

figure;
imshow(uint8(new_image_r));
end