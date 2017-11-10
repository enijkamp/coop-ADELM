net = [];
config = frame_config('face10000', 'dense', 'mask');
net.normalization.imageSize = [128,128,3];
config.sx = 128;
config.sy = 128;
[imdb, getBatch, net] = create_imdb(config, net);

mask_sz = 60;
dst = ['../../data/brick1_', num2str(mask_sz), '/'];

if ~exist(dst, 'dir')
   mkdir(dst); 
end

train = find(imdb.images.set==1);
val = find(imdb.images.set==2) ; 


masks = generate_masks([128,128,3,numel(train) + numel(val)], mask_sz);
save([dst,'masks.mat'], 'masks');

for t=1:config.BatchSize:numel(train)
    fprintf('batch %3d/%3d: \n', ...
        fix(t/config.BatchSize)+1, ceil(numel(train)/config.BatchSize)) ;
    batchSize = min(config.BatchSize, numel(train) - t + 1) ;
    
    batchStart = t;
    batchEnd = min(t+config.BatchSize-1, numel(train)) ;
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

for t=1:config.BatchSize:numel(val)
    fprintf('batch %3d/%3d: \n', ...
        fix(t/config.BatchSize)+1, ceil(numel(val)/config.BatchSize)) ;
    batchSize = min(config.BatchSize, numel(val) - t + 1) ;
    
    batchStart = t;
    batchEnd = min(t+config.BatchSize-1, numel(val)) ;
    batch = val(batchStart : batchEnd) ;
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