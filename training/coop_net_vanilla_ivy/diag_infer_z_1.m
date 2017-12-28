function [] = diag_infer_z_1(pool_name, dirs)

%% args
if nargin < 1
    pool_name = 'local';
    dirs = { 'ivy_dense_em_11_11' };
end

%% prep

% clear
close all;
restoredefaultpath();

% cpu
exp_id = 1;

% prep
learningTime = tic;
%diary(['diag_infer_z_' num2str(exp_id) '.out']);
set(0,'DefaultTextInterpreter','none');
addpath(genpath('./core/'));
addpath(genpath('../../main/'));
Setup_CPU(false);
rng(123);

% parpool
no_workers = 24;
if no_workers > 1
    delete(gcp('nocreate'));
    parpool(pool_name, no_workers, 'IdleTimeout', Inf);
end

% set matconvnet for each worker
if no_workers > 1
    parfor i = 1:no_workers
        vl_setupnn();
    end
end


%% load

for exp_i = 1:length(dirs)
    
    disp(['### ' num2str(exp_i) '/' num2str(length(dirs)) ' ###']);

    net_id = dirs{exp_i};
    exp_path = [ 'working/' net_id '/' ];

    if exist([exp_path 'layer_01_iter_2200_model.mat'], 'file')
        loaded_nets = load([exp_path 'layer_01_iter_2200_model.mat']);
    elseif exist([exp_path 'layer_01_iter_1200_model.mat'], 'file')
        loaded_nets = load([exp_path 'layer_01_iter_1200_model.mat']);
    elseif exist([exp_path 'layer_01_iter_400_model.mat'], 'file')
        loaded_nets = load([exp_path 'layer_01_iter_400_model.mat']);
    elseif exist([exp_path 'layer_01_iter_200_model.mat'], 'file')
        loaded_nets = load([exp_path 'layer_01_iter_200_model.mat']);
    elseif exist([exp_path 'layer_01_iter_100_model.mat'], 'file')
        loaded_nets = load([exp_path 'layer_01_iter_100_model.mat']);
    elseif exist([exp_path 'layer_01_iter_32_model.mat'], 'file')
        loaded_nets = load([exp_path 'layer_01_iter_32_model.mat']);
    else
        continue;
    end

    loaded_config = load([exp_path 'config.mat']);
    config_train = loaded_config.config;

    %% config

    % nets
    config.des_net = loaded_nets.net1;
    config.gen_net = loaded_nets.net2;

    % image and latent space
    config.z_sz = [1,1,30];
    config.im_sz = [64,64,3];
    config.mean_im = single(zeros(config.im_sz));

    % params
    config.Delta2 = 0.05;
    config.refsig2 = config_train.refsig2;


    %% inference

    % images
    files = strings(2000,1);
    for i = 1:2000
       files(i) =  ['../data/ivy/all/' num2str(i) '.png'];
    end
    images_train = read_images('../data/ivy/all/', files, [64, 64, 3]);

    % results
    images_min_dist = zeros(size(images_train,4), 1);
    images_min_dist_index = zeros(size(images_train,4), 1);
    images_zs = zeros(1,1,30,size(images_train,4));

    % infer
    parfor i = 1:size(images_train,4)
        for_time = tic;

        % images
        syn_mat = images_train(:,:,:,i);
        gen_im = floor((syn_mat+1)*128);

        % langevin
        z = randn(config.z_sz,'single');
        t_max = 1000;
        dists = zeros(t_max, 1);
        z_min = zeros(config.z_sz);
        dist_min = inf;
        for t = 1:t_max
            res = vl_gan_cpu(config.gen_net, z, syn_mat, [], 'conserveMemory', 1);
            delta_log = res(1).dzdx / config.refsig2 / config.refsig2 - z;
            z = z + config.Delta2 * config.Delta2 / 2 * delta_log;
            %z = z + config.Delta2 * randn(size(z2), 'single');

            gen_im_z = floor((res(end).x+1)*128);
            dists(t,1) = sqrt(sum((gen_im(:)-gen_im_z(:)).^2));

            if dists(t,1) < dist_min
                z_min = z;
                dist_min = dists(t,1);
            end
        end

        % store
        [images_min_dist(i), images_min_dist_index(i)] = min(dists);
        images_zs(:,:,:,i) = z_min;

        fprintf('%4.0f / %4.0f -> %4.4f %4.0f (%3.0fs)\n', i, size(images_train,4), images_min_dist(i), images_min_dist_index(i), toc(for_time));
    end

    % save
    if exist(['diag/infer_z_' num2str(exp_id) '/' net_id], 'dir')
        rmdir(['diag/infer_z_' num2str(exp_id) '/' net_id], 's');
    end
    mkdir(['diag/infer_z_' num2str(exp_id) '/' net_id]);

    f1 = figure;
    hist(images_min_dist, 40);
    saveas(f1, ['diag/infer_z_' num2str(exp_id) '/' net_id '/hist.fig']);
    saveas(f1, ['diag/infer_z_' num2str(exp_id) '/' net_id '/hist.png']);

    f2 = figure;
    hold on;
    plot(1:500, images_min_dist(1:500), 'r.');
    plot(501:1000, images_min_dist(501:1000), 'b.');
    plot(1001:1500, images_min_dist(1001:1500), 'k.');
    plot(1501:2000, images_min_dist(1501:2000), 'm.');
    hold off;
    saveas(f2, ['diag/infer_z_' num2str(exp_id) '/' net_id '/dists.fig']);
    saveas(f2, ['diag/infer_z_' num2str(exp_id) '/' net_id '/dists.png']);

    save(['diag/infer_z_' num2str(exp_id) '/' net_id '/images_min_dist.mat'], 'images_min_dist');
    save(['diag/infer_z_' num2str(exp_id) '/' net_id '/images_min_dist_index.mat'], 'images_min_dist_index');
    save(['diag/infer_z_' num2str(exp_id) '/' net_id '/images_zs.mat'], 'images_zs');

end

% time
learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);

disp('done');

end

function [img_mat] = read_images(inPath, files, imageSize)

if isempty(files)
    fprintf(['error: No training images are found in "' inPath '"\n']);
    keyboard;
end

img_mat = zeros([imageSize, length(files)], 'single');
for iImg = 1:length(files)
    img = single(imread(char(files(iImg))));
    img = imresize(img, imageSize(1:2));
    min_val = min(img(:));
    max_val = max(img(:));
    img_mat(:,:,:,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
end

end