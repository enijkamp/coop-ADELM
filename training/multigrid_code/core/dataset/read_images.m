function [img_mat, net] = read_images(config, net)

img_file = [config.working_folder, 'images.mat'];
files = dir([config.inPath '*']);
large = 0;
if isempty(files)
    fprintf('error: No training images are found\n');
    keyboard;
else
    files = dir([config.inPath '*.jpg']);
    if isempty(files)
        files = dir([config.inPath '*.JPEG']); 
    end
    if isempty(files)
        files = dir([config.inPath '*.png']); 
    end
    if isempty(files)
        category_lists = dir([config.inPath, '*']);
        valid_category = true(1, length(category_lists));
        for i = 1:length(category_lists)
            if category_lists(i).name(1) == '.' || ~category_lists(i).isdir
               valid_category(i) = false; 
            end
        end
        category_lists = category_lists(valid_category);
        large = 1;
    end
end


if isfield(config, 'num_img')
    num_img = config.num_img;
else
    num_img = length(files);
end

if large == 0
    img_mat = zeros([64, 64, 3, num_img], 'single');
    for iImg = 1:num_img
        fprintf('read and process images %d / %d\n', iImg, num_img)
        img = imread(fullfile(config.inPath, files(iImg).name));
        img = single(imresize(img, [64,64]));
        if length(size(img)) == 2
            img_mat(:,:,1,iImg) = img;
            img_mat(:,:,2,iImg) = img;
            img_mat(:,:,3,iImg) = img;
        else
            img_mat(:,:,:,iImg) = img;
        end
    end
    save(img_file, 'img_mat');
end

if large == 1
    labels = [];
    img_mat = zeros([64, 64, 3, num_img * length(category_lists)], 'single');
    for ii = 1:length(category_lists)
        inPath = [config.inPath, category_lists(ii).name,'/'];
        files = dir([inPath '*.jpg']);
        if isempty(files)
           files = dir([inPath '*.JPEG']); 
        end
        if isempty(files)
           files = dir([inPath '*.png']); 
        end
        
        for iImg = 1:num_img
            fprintf('read and process images in %s: %d / %d\n', category_lists(ii).name, iImg, num_img)
            img = imread(fullfile(inPath, files(iImg).name));
            img = single(imresize(img, [64,64]));
            if length(size(img)) == 2
                img_mat(:,:,1, (ii-1)*num_img+iImg) = img;
                img_mat(:,:,2,(ii-1)*num_img+iImg) = img;
                img_mat(:,:,3,(ii-1)*num_img+iImg) = img;
            else
                img_mat(:,:,:,(ii-1)*num_img+iImg) = img;
            end
        end  
        labels = [labels, ones(1, num_img)*ii];
    end
    index = randperm(size(img_mat,4))
    img_mat = img_mat(:,:,:, index);
    labels = labels(index);
    save(img_file, 'img_mat', 'labels');
end