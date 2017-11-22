%% prepare images
% category1 = 'scene';
% category = 'scene_denoise';
% model1_path = ['/media/vclagpu/Data1/ruiqi/workingspace/multigrid_v2/results/', category1, '_4to16_701.mat'];
% model2_path = ['/media/vclagpu/Data1/ruiqi/workingspace/multigrid_v2/results/', category1, '_16to64_271.mat'];
% load(model2_path);
% config = frame_config('scene');

% batchSize = opts.batch_size ;
% batchTime = tic ;
% config = opts;
% 
% [imdb, getBatch, net] = create_imdb(config, net3);
% subset = find(imdb.images.set == 1);
% build filter Q
Q = zeros(4, 4, 3, 'single');
Q(:,:,1,1) = 1/16;
Q(:,:,2,2) = 1/16;
Q(:,:,3,3) = 1/16;
Q = gpuArray(Q);

% build filter Q^(-1)
Q_inv = zeros(4, 4, 3, 'single');
Q_inv(:,:,1,1) = 1;
Q_inv(:,:,2,2) = 1;
Q_inv(:,:,3,3) = 1;
Q_inv = gpuArray(Q_inv);

Q_dummy = zeros(1, 3, 'single');
Q_dummy = gpuArray(Q_dummy);

t = 1;
% get this image batch and prefetch the next
batchStart = t;
batchEnd = min(t+opts.batch_size-1, numel(subset)) ;
batch = subset(batchStart : batchEnd) ;
im = getBatch(imdb, batch) ;
im = gpuArray(im);

% construct y^(0)
y0 = vl_nnconv(im, Q, Q_dummy, 'stride', 4, 'CuDNN') ;
y0 = vl_nnconv(y0, Q, Q_dummy, 'stride', 4, 'CuDNN') ;
y0 = vl_nnconv(y0, Q, Q_dummy, 'stride', 4, 'CuDNN') ;
y0 = vl_nnconvt(y0, Q_inv, Q_dummy, 'upsample', 4, 'CuDNN') ;
%     y0 = imresize(imresize(im, [opts.sx/2, opts.sy/2], 'box'), [opts.sx, opts.sy], 'nearest');
syn_mat = y0;

%% 4 to 16
opts.Synfolder = './results/';
net1 = vl_simplenn_move(net1, 'gpu') ;

res = [];
loss = 0;
mean_img1 = ones(net1.normalization.imageSize) * net.normalization.averageImage(1,1,1,1);
for t=1:1  
    
    % prepare derative
    numImages = size(im, 4);
    dydz = gpuArray(ones(net1.dydz_sz, 'single'));
    dydz = repmat(dydz, 1, 1, 1, numImages);      
    
    for tt = 1:opts.T
        res = vl_simplenn(net1, syn_mat, dydz, res, ...
            'accumulate', false, ...
            'disableDropout', false, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn);
        
        % y^(t+1)
%         syn_mat = syn_mat +  opts.Delta^2/2 * (res(1).dzdx - syn_mat / opts.refsig^2)+ ...
%             opts.Delta * gpuArray(randn(size(syn_mat), 'single'));
        syn_mat = syn_mat +  opts.Delta^2/2 * (res(1).dzdx - syn_mat / opts.refsig^2);
        
%         y_sub = vl_nnconv(syn_mat, Q, Q_dummy, 'stride', 2, 'CuDNN');
%         y_sub = vl_nnconvt(y_sub, Q_inv, Q_dummy, 'upsample', 2, 'CuDNN') ;
%         syn_mat = y0 + syn_mat - y_sub;
    end
    
    draw_figures(opts, gather(syn_mat), 0, mean_img1, [], '1to4_recovered');
    
   % loss = loss + gather( mean(reshape(abs(syn_mat - im), [], 1))) /1 / 2;
end

net1 = vl_simplenn_move(net1, 'cpu') ;


%% 16 to 64 (after 4 to 16)
net2 = vl_simplenn_move(net2, 'gpu') ;
mean_img2 = ones(net2.normalization.imageSize) * net.normalization.averageImage(1,1,1,1);
res = [];

syn_mat = vl_nnconvt(syn_mat, Q_inv, Q_dummy, 'upsample', 4, 'CuDNN') ;
numImages = size(syn_mat, 4);
dydz = gpuArray(ones(net2.dydz_sz, 'single'));
dydz = repmat(dydz, 1, 1, 1, numImages);
cell_idx = (ceil(t / opts.batch_size) - 1) * numlabs + labindex;   

for tt = 1:30
    res = vl_simplenn(net2, syn_mat, dydz, res, ...
        'accumulate', false, ...
        'disableDropout', false, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn);

    % y^(t+1)
%     syn_mat = syn_mat +  opts.Delta^2/2 * (res(1).dzdx - syn_mat / opts.refsig^2)+ ...
%         opts.Delta * gpuArray(randn(size(syn_mat), 'single'));
    syn_mat = syn_mat +  opts.Delta^2/2 * (res(1).dzdx - syn_mat / opts.refsig^2);
end

draw_figures(opts, gather(syn_mat), 0, mean_img2, [],  '4to16_recovered');

net2 = vl_simplenn_move(net2, 'cpu') ;
 
%% 16 to 64 (from original 16 to 64)
net3 = vl_simplenn_move(net3, 'gpu') ;
mean_img3 = ones(net3.normalization.imageSize) * net.normalization.averageImage(1,1,1,1);
res = [];
 % construct y^(0)
syn_mat = vl_nnconvt(syn_mat, Q_inv, Q_dummy, 'upsample', 4, 'CuDNN') ;
numImages = size(syn_mat, 4);
dydz = gpuArray(ones(net3.dydz_sz, 'single'));
dydz = repmat(dydz, 1, 1, 1, numImages);
cell_idx = (ceil(t / opts.batch_size) - 1) * numlabs + labindex;   

for tt = 1:30
    res = vl_simplenn(net3, syn_mat, dydz, res, ...
        'accumulate', false, ...
        'disableDropout', false, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn);

    % y^(t+1)
%     syn_mat = syn_mat +  opts.Delta^2/2 * (res(1).dzdx - syn_mat / opts.refsig^2)+ ...
%         opts.Delta * gpuArray(randn(size(syn_mat), 'single'));
    syn_mat = syn_mat +  opts.Delta^2/2 * (res(1).dzdx - syn_mat / opts.refsig^2);
end

draw_figures(opts, gather(syn_mat), 0, mean_img3, [], '16to64_recovered');

net3 = vl_simplenn_move(net3, 'cpu') ;
 
 
 