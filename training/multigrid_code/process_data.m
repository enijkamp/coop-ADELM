net = [];
config = frame_config('face1000', 'mask');
img_sz = 64;
net.normalization.imageSize = [img_sz,img_sz,3];
config.sx = img_sz;
config.sy = img_sz;
[imdb, getBatch, net] = create_imdb(config, net);

ratio = 0.3;
dst = sprintf('../../data/face1000_%.1f/', ratio);

% dst = '../../data/face10000_ori/';

if ~exist(dst, 'dir')
   mkdir(dst); 
end

train = find(imdb.images.set==1);
val = find(imdb.images.set==2) ; 

masks = generate_salt_pepper([img_sz, img_sz, 3, numel(train) + numel(val)], ratio, 3);
save([dst,'masks.mat'], 'masks');

for t=1:config.batch_size:numel(train)
    fprintf('batch %3d/%3d: \n', ...
        fix(t/config.batch_size)+1, ceil(numel(train)/config.batch_size)) ;
    batchSize = min(config.batch_size, numel(train) - t + 1) ;
    
    batchStart = t;
    batchEnd = min(t+config.batch_size-1, numel(train)) ;
    batch = train(batchStart : batchEnd) ;
    im = getBatch(imdb, batch) ;
    mask = masks(:,:,:,batch);
    im(mask) = 0;
    
    for i = 1:numel(batch)
        img = im(:,:,:,i);
        gLow = min( reshape(img, [],1));
        gHigh = max(reshape(img, [],1));
        img = (img-gLow) / (gHigh - gLow);
        imwrite(img, [dst, sprintf('%06d.jpg', batch(i))]);
    end
end

% for t=1:config.batch_size:numel(val)
%     fprintf('batch %3d/%3d: \n', ...
%         fix(t/config.batch_size)+1, ceil(numel(val)/config.batch_size)) ;
%     batchSize = min(config.batch_size, numel(val) - t + 1) ;
%     
%     batchStart = t;
%     batchEnd = min(t+config.batch_size-1, numel(val)) ;
%     batch = val(batchStart : batchEnd) ;
%     im = getBatch(imdb, batch) ;
%     mask = masks(:,:,:,batch);
%     im(mask) = 0;
%     
%     for i = 1:numel(batch)
%         img = im(:,:,:,i);
%         gLow = min( reshape(img, [],1));
%         gHigh = max(reshape(img, [],1));
%         img = (img-gLow) / (gHigh - gLow);
%         imwrite(img, [dst, sprintf('%06d.jpg', batch(i))]);
%     end
% end