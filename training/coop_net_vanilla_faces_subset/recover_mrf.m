function error = recover_mrf(im, im_ori, masks, save_dir, method)
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
im = gather(im);
im_ori = gather(im_ori);
masks = gather(masks);

num_sweep = 500;
error = 0;

if method == 1
    beta = 0.0005;
else
    beta = sqrt(0.0005);
end

for i = 1:size(im, 4)
    fprintf('MRF recovery, image #%d\n', i);
    img = (im(:,:,:,i)+1)/2 * 255;
    img = padarray(img, [1, 1]);
    mask = masks(:,:,1, i);
    mask = padarray(mask, [1, 1]);
    
    img_ori = im_ori(:,:,:,i);
    img_ori = padarray(img_ori, [1, 1]);
    idx = find(mask == 1);
    [idx_r, idx_c] = ind2sub(size(mask), idx);
    for t = 1:num_sweep
        for ch = 1:size(img, 3)
            R_I = img(:,:,ch);
            for j = 1:numel(idx_r)
               r = idx_r(j);
               c = idx_c(j);
               R_I(r,c) = Gibbs(r, c, R_I, beta, method);
            end
            img(:,:,ch) = R_I;
        end
    end
    
    img = img/255 * 2 -1;
    error = error + mean(reshape(abs(img(mask) - img_ori(mask)), [], 1)) / size(im, 4) / 2;
    
    img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
    imwrite(img, [save_dir, num2str(i, 'recover_%04d.png')]);
end

function [pix]=Gibbs(x,y,im,beta,method)
% pix= 0-255
pix = -1;

i=0:1:255;

switch method
    case 1
        X=exp (-beta* ( (im(x+1,y)-i).^2 + (im(x,y+1)-i).^2 + (im(x-1,y)-i).^2 + (im(x,y-1)-i).^2 + (im(x-1,y-1)-i).^2 + (im(x-1,y+1)-i).^2 + (im(x+1,y-1)-i).^2 + (im(x+1,y+1)-i).^2 ));
    case 2
        X=exp (-beta* ( abs(im(x+1,y)-i) + abs(im(x,y+1)-i) + abs(im(x-1,y)-i) + abs(im(x,y-1)-i) + abs(im(x-1,y-1)-i) + abs(im(x-1,y+1)-i) + abs(im(x+1,y-1)-i) + abs(im(x+1,y+1)-i) ));
    otherwise
        error('No such a method');
end
z=sum(X);
X=X./z;
CDF_X=cumsum(X);

ra=rand();
for i=1:256
    if ra<=CDF_X(i)
        pix=i-1;
        break;
    end
end