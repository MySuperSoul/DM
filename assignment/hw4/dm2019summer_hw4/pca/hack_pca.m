function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation

img_r = double(imread(filename));

% YOUR CODE HERE
% here use position in image to represent features
% filter non background piexls
[x, y] = find(img_r < 255);
data = [x, y];
[eigvectors, eigvalues] = pca(data);
project_pos = data * eigvectors;
project_pos = ceil(project_pos);
% normalize
project_pos = project_pos - min(project_pos, [], 1) + 1;
new_image = ones(size(img_r)) * 255;

for i=1:size(project_pos, 1)
    row_origin = data(i, :);
    row_project = project_pos(i, :);
    new_image(row_project(2), row_project(1)) = img_r(row_origin(1), row_origin(2));
end
new_image = flipud(uint8(new_image));
figure;
imshow(new_image);
end