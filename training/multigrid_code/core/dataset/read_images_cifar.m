function [img_mat, net] = read_images_cifar(config, net)

img_mat = [];
Labels = [];
% for ii = 1:5
%     load(['/media/vclagpu/Data1/ruiqi/image/cifar-10-batches-mat/data_batch_',num2str(ii),'.mat']);
%     data = permute(data, [2 1]);
%     data = reshape(data, [32 32 3 size(data, 2)]);
%     data = permute(data, [2 1 3 4]);
%     img_mat = cat(4, img_mat, data);
%     Labels = [Labels; labels];
% end



load('/media/vclagpu/Data1/ruiqi/image/cifar-10-batches-mat/test_batch.mat');
data = permute(single(data), [2 1]);
data = reshape(data, [32 32 3 size(data, 2)]);
data = permute(data, [2 1 3 4]);
img_mat = cat(4, img_mat, data);
Labels = [Labels; labels];

Img_mat = zeros([64, 64, 3, size(img_mat, 4)]);
for i = 1:size(img_mat, 4)
    Img_mat(:,:,:,i) = single(imresize(img_mat(:,:,:,i), [64 64]));
    if mod(i, 1000) == 0
        disp(i)
    end
end

% img_mat = img_mat(:,:,:,1:200);

num_img = size(img_mat, 4);
mean_img = mean(img_mat(:));
disp(mean_img);

if ~isfield(net.normalization, 'averageImage')
    net.normalization.averageImage = ones(net.normalization.imageSize) ...
        * mean_img;
end


for iImg = 1:num_img
    img_mat(:,:,:,iImg) = img_mat(:,:,:,iImg) - net.normalization.averageImage;
end
