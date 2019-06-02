K = 5;
for i=1:K
    figure;
    img_name = strcat('./TestSet/test_', num2str(i));
    img_name = strcat(img_name, '.gif');
    digits = hack(img_name);
    show_image(extract_image(img_name));
    title(num2str(digits));
    hold on;
end
hold off;