function [img_mats] = read_images_all(num_imgs)

if nargin < 1
    num_imgs = 97;
end

imageSize = [64, 64, 3];

category_path = '../data/faces_subset_dog/';
category_lists = dir([category_path, '*']);
valid_category = true(1, length(category_lists));
for i = 1:length(category_lists)
    if category_lists(i).name(1) == '.' || ~category_lists(i).isdir
       valid_category(i) = false; 
    end
end

category_lists = category_lists(valid_category);

img_mats = [];

for ii = 1:length(category_lists)
    category = category_lists(ii).name;
    inPath = [category_path, category,'/'];
    
    files = dir([inPath '*.jpg']);

    if isempty(files)
       files = dir([inPath '*.JPEG']); 
    end
    if isempty(files)
       files = dir([inPath '*.png']); 
    end

    if isempty(files)
        fprintf('error: No training images are found\n');
        keyboard;
    end    
    disp(category);
    disp(num2str(length(files)));
    num_imgs = length(files);
    files = files(1:num_imgs);
    img_mat = zeros([imageSize, length(files)], 'single');
    for iImg = 1:length(files)
        fprintf('read and process images in %s %d / %d\n', category, iImg, length(files))
        img = single(imread(fullfile(inPath, files(iImg).name)));
        
        % center crop human heads
        if strcmp(category, 'HumanHead') && false
            [p3, p4, ~] = size(img);
            q1 = 160;
            i3_start = floor((p3-q1)/2);
            i3_stop = i3_start + q1;

            i4_start = floor((p4-q1)/2);
            i4_stop = i4_start + q1;

            img = img(i3_start:i3_stop, i4_start:i4_stop, :);
            figure, imshow(uint8(img));
        end
        
        img = imresize(img, imageSize(1:2));
        if iImg > 1 && iImg < 4 && false
            figure, imshow(uint8(img));
        end
        min_val = min(img(:));
        max_val = max(img(:));
        if length(size(img)) == 2
            img_mat(:,:,1,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
            img_mat(:,:,2,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
            img_mat(:,:,3,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;          
        else
            img_mat(:,:,:,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
        end
    end
    img_mats = cat(4, img_mats, img_mat);  
end

img_mats = img_mats(:,:,:,randperm(size(img_mats,4)) );

end







    


