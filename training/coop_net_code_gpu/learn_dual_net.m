function [net1,net2,gen_mats,syn_mats] = learn_dual_net(config,net1,net2,fix_des,fix_gen)
    if nargin < 4 || isempty(fix_des), fix_des = 0; end
    if nargin < 5 || isempty(fix_gen), fix_gen = 0; end
    learningTime = tic;
    
    % read in config
    if nargin < 1 || isempty(config), config = train_coop_config; end
    
    % gpu
    global use_gpu;
    use_gpu = config.use_gpu;
    
    if use_gpu
        parallel.gpu.rng(0, 'Philox4x32-10');
    end
    
    % generator net2
    if nargin < 3 || isempty(net2)
        rng(123);
        net2 = mnist_frame_gan_params();
    end

    % descriptor net1
    if nargin < 2 || isempty(net1)
        %[~,net1] = train_coop_config;
        net1 = [];
        net1.layers = {};
        rng(123);
        net1 = mnist_add_bottom_filters(net1, 1:3, config);
        net1.mean_im = config.mean_im;
    end
 
    % generator net config
    config.z_sz = [1, 1, size(net2.layers{1}.weights{1}, 4)];
    net2.z_sz = config.z_sz;
    config.dydz_sz2 = [config.z_sz(1:2), 1];
    for l = 1:numel(net2.layers)
        if strcmp(net2.layers{l}.type, 'convt')
            f_sz = size(net2.layers{l}.weights{1});
            crops = [net2.layers{l}.crop(1)+net2.layers{l}.crop(2), ...
                net2.layers{l}.crop(3)+net2.layers{l}.crop(4)];
            config.dydz_sz2(1:2) = net2.layers{l}.upsample.*(config.dydz_sz2(1:2) - 1) ...
                + f_sz(1:2) - crops;
        end
    end
    net2.dydz_sz = config.dydz_sz2;

    z = randn(config.z_sz, 'single');
    if config.use_gpu
        net2 = vl_simplenn_move(net2, 'gpu');
        res = vl_gan(net2, gpuArray(z));
        net2 = vl_simplenn_move(net2, 'cpu');
    else
        res = vl_gan_cpu(net2, z);
    end
    
    net2.numFilters = zeros(1, length(net2.layers));
    for l = 1:length(net2.layers)
        if isfield(net2.layers{l}, 'weights')
            sz = size(res(l+1).x);
            net2.numFilters(l) = sz(1) * sz(2);
        end
    end

    config.layer_sets2 = numel(net2.layers):-1:1;

    net2.normalization.imageSize = config.dydz_sz2;
    net2.normalization.averageImage = zeros(config.dydz_sz2, 'single');
    config.sx = config.dydz_sz2(1);
    config.sy = config.dydz_sz2(2);
    clear z;
    
    % descriptor net config
    img = randn([config.im_size,config.im_size,3],'single');
    if config.use_gpu
        net1 = vl_simplenn_move(net1, 'gpu') ;
        res = vl_simplenn(net1, gpuArray(img));
        net1 = vl_simplenn_move(net1, 'cpu');
    else
        res = vl_simplenn(net1, img);
    end
    config.dydz_sz1 = size(res(end).x);
    net1.numFilters = zeros(1, length(net1.layers));
    for l = 1:length(net1.layers)
        if isfield(net1.layers{l}, 'weights')
          sz = size(res(l+1).x);
          net1.numFilters(l) = sz(1) * sz(2);
        end
    end
    net1.dydz_sz = config.dydz_sz1;

    config.layer_sets1 = numel(net1.layers):-1:1;
    
    %% print networks
    print_nets(net1, net2, config.im_size, config.z_sz);
    
    %% print data
    config.nTileRow = uint32(sqrt(config.num_syn));
    config.nTileCol = uint32(sqrt(config.num_syn));
    [I_imdb, ~] = convert_syns_mat(config, net1.mean_im, config.imdb);
    imwrite(I_imdb, [config.trained_folder, 'data', '.png']);

    clear res;
    clear img;
    
    %training options
    opts.conserveMemory = true ;
    opts.backPropDepth = +inf ;
    opts.sync = false ;
    opts.prefetch = false;
    opts.cudnn = true ;
    opts.weightDecay = 0.0001 ;
    opts.momentum = 0.5;

    net1 = initialize_momentum(net1);
    net2 = initialize_momentum(net2);

    for i=1:numel(net1.layers)
        if isfield(net1.layers{i}, 'weights')
            J = numel(net1.layers{i}.weights);
            for j=1:J
                net1.layers{i}.momentum{j} = zeros(size(net1.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net1.layers{i}, 'learningRate')
                net1.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net1.layers{i}, 'weightDecay')
                net1.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end

    net1.filterSelected = 1:prod(config.dydz_sz1);
    net1.selectedLambdas = ones(1, prod(config.dydz_sz1), 'single');
    
    if config.use_gpu
        numGpus = 1;
        gpuDevice(numGpus);
    end

    % -------------------------------------------------------------------------
    %                           Train and validate
    % -------------------------------------------------------------------------

    delete([config.gen_im_folder,'*.png']);
    delete([config.syn_im_folder,'*.png']);
    
    save([config.trained_folder,'config.mat'],'config');
    
    loss1 = [];
    loss2 = [];
    
    rng(123);
    h = figure;
    for epoch=1:config.nIteration

        [net1_out,net2_out,gen_mats,syn_mats,z,loss2(epoch)] = process_epoch_dual(opts,epoch,net1,net2,config);     

        if ~fix_des, net1 = net1_out; end
        if ~fix_gen, net2 = net2_out; end
        
        % loss
        % loss 1: l2 distance synthesis and revision (revision should be small)
        % loss 2: mean of gradients of descriptor (if training equals synthesis, then gradient should be 0)
        loss1(epoch) = compute_loss(opts, syn_mats, net2_out, z, config);
        save([config.trained_folder,'loss.mat'],'loss1','loss2');
        fprintf('loss1 = %.4f.\nloss2 = %.4f.\n', loss1(epoch), loss2(epoch));
        clf(h);
        hold on;
        plot(loss1);
        plot(loss2);
        hold off;
        drawnow;
        
        % synthesis
        if mod(epoch - 1, 10) == 0 || epoch == config.nIteration
            gen_mats = generate_imgs(opts, net2, z);
            gen_mats = floor(128*(gen_mats+1)) - repmat(config.mean_im, 1, 1, 1, config.num_syn);
            config.nTileRow = uint32(sqrt(config.num_syn));
            config.nTileCol = uint32(sqrt(config.num_syn));
            draw_figures(config, config.syn_im_folder, syn_mats, epoch, config.mean_im, 'net1');
            draw_figures(config, config.gen_im_folder, gen_mats, epoch, config.mean_im, 'net2');
        end
    end
    
    learningTime = toc(learningTime);
    save([config.trained_folder,'time.mat'],'learningTime');
    
    hrs = floor(learningTime / 3600);
    learningTime = mod(learningTime, 3600);
    mins = floor(learningTime / 60);
    secds = mod(learningTime, 60);
    fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);
end

function imgs = generate_imgs(opts, net_cpu, z)
net = vl_simplenn_move(net_cpu, 'gpu') ;
res = vl_gan(net, gpuArray(z), [], [], ...
    'accumulate', false, ...
    'disableDropout', true, ...
    'conserveMemory', opts.conserveMemory, ...
    'backPropDepth', opts.backPropDepth, ...
    'sync', opts.sync, ...
    'cudnn', opts.cudnn) ;
imgs = gather(res(end).x);
end

function [] = draw_figures(config, out_dir, syn_mat, iter, mean_img, prefix)
[I_syn, ~] = convert_syns_mat(config, mean_img, syn_mat);
imwrite(I_syn, [out_dir, prefix, num2str(iter, '_%04d'), '.png']);
end

function [I_syn, syn_mat] = convert_syns_mat(config, mean_img, syn_mat)
space = 5;
color = 0;
for i = 1:size(syn_mat, 4)
    syn_mat(:,:,:,i) = uint8(syn_mat(:,:,:,i) + mean_img);
end
I_syn = mat2canvas(syn_mat, config, space);
for row = 1:config.nTileRow-1
    I_syn(row * config.sx + (row-1) * space + 1:row * config.sx + (row-1) * space + space, :, :) = color;
end
for col = 1:config.nTileCol-1
    I_syn(:, col * config.sy + (col-1) * space + 1:col * config.sy + (col-1) * space + space, :) = color;
end
I_syn = uint8(I_syn);
syn_mat = uint8(syn_mat);
end

function I_syn = mat2canvas(syn_mat, config, space)
sx = config.sx;
sy = config.sy;
dim3 = size(syn_mat, 3);
num_syn = size(syn_mat, 4);
I_syn = zeros(config.nTileRow * config.sy + space * (config.nTileRow-1), ...
    config.nTileCol * config.sx + space * (config.nTileCol-1), dim3, 'single');
k = 1;
for j = 1:config.nTileRow
    for i = 1:config.nTileCol
        I_syn( 1+(j-1)*sy + (j-1) * space : sy+(j-1)*sy + (j-1) * space, 1+(i-1)*sx + (i-1)*space : sx+(i-1)*sx + (i-1) * space, :) = syn_mat(:,:,:,k);
        k = k+1;
        if k > num_syn
            return;
        end
    end
end
end

function loss = compute_loss(opts, syn_mat, net2_cpu, z, config)
if config.use_gpu
    net2 = vl_simplenn_move(net2_cpu, 'gpu');
    res = [];
    res = vl_gan(net2, gpuArray(z), gpuArray(syn_mat), res, ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    loss = gather( mean(reshape(sqrt((res(end).x - syn_mat).^2), [], 1)));
else
    res = [];
    res = vl_gan_cpu(net2_cpu, z, syn_mat, res, ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    loss = gather( mean(reshape(sqrt((res(end).x - syn_mat).^2), [], 1)));
end
end

function net = initialize_momentum(net)
for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights) ;
        for j=1:J
            net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
        end
        if ~isfield(net.layers{i}, 'learningRate')
            net.layers{i}.learningRate = ones(1, J, 'single') ;
        end
        if ~isfield(net.layers{i}, 'weightDecay')
            net.layers{i}.weightDecay = ones(1, J, 'single') ;
        end
    end
end
end
