function [] = diag_infer_z_2()

%% prep

% clear
clear all;
close all;
restoredefaultpath();

% gpu
exp_id = 2;

%% issue: Generator batch normalization in langevin. We can't reach single image.

%% cpu  (1 image) -> 10s per image
%    1 / 2000 -> 8324.2627  936 (10.66s)
%    2 / 2000 -> 4791.4805  932 (10.36s)
%    3 / 2000 -> 4815.6265  674 (10.79s)
%    4 / 2000 -> 8953.1074  642 (10.14s)
%    5 / 2000 -> 5060.9888 1000 (10.16s)
%    6 / 2000 -> 4786.8652  996 (10.09s)
%    7 / 2000 -> 8340.2031  952 (10.10s)
%    8 / 2000 -> 5066.5166  977 (10.14s)
%    9 / 2000 -> 5010.1006  868 (10.20s)
%   10 / 2000 -> 5138.6963  930 (10.20s)

%% gpu (1 image) -> 3.5s per image (cudnn)
%    1 / 2000 -> 8272.9805  606 (5.36s)
%    2 / 2000 -> 4795.9683  965 (3.47s)
%    3 / 2000 -> 8652.7363  820 (3.44s)
%    4 / 2000 -> 4758.5669  689 (3.44s)
%    5 / 2000 -> 5054.4019  998 (3.44s)
%    6 / 2000 -> 4655.9341  921 (3.44s)
%    7 / 2000 -> 4939.8921 1000 (3.44s)
%    8 / 2000 -> 8584.7168  636 (3.48s)
%    9 / 2000 -> 4993.5620  983 (3.49s)
%   10 / 2000 -> 5245.5996 1000 (3.43s)

%% gpu (1 image) -> 4s per image (no cudnn)
%    1 / 2000 -> 8272.9805  606 (5.60s)
%    2 / 2000 -> 4795.9683  965 (3.94s)
%    3 / 2000 -> 8652.7363  820 (3.91s)
%    4 / 2000 -> 4758.5669  689 (3.91s)
%    5 / 2000 -> 5054.4019  998 (3.90s)
%    6 / 2000 -> 4655.9341  921 (3.91s)
%    7 / 2000 -> 4939.8921 1000 (4.13s)
%    8 / 2000 -> 8584.7168  636 (3.94s)
%    9 / 2000 -> 4993.5620  983 (3.91s)
%   10 / 2000 -> 5245.5996 1000 (3.92s)

%% gpu (100 images) -> 0.9s per image -> ISSUE bnorm
%    1 / 2000 -> 4138.5938  993 (  1s)
%    2 / 2000 -> 4283.5366  999 (  1s)
%    3 / 2000 -> 4357.0054  995 (  1s)
%    4 / 2000 -> 4503.8530  624 (  1s)
%    5 / 2000 -> 4496.9321  951 (  1s)
%    6 / 2000 -> 4230.5083  950 (  1s)
%    7 / 2000 -> 4231.3018  995 (  1s)
%    8 / 2000 -> 4581.1431  785 (  1s)
%    9 / 2000 -> 4572.6948  999 (  1s)
%   10 / 2000 -> 4527.2324  529 (  1s)

use_gpu = 1;

% prep
learningTime = tic;
diary(['diag_infer_z_' num2str(exp_id) '.out']);
set(0,'DefaultTextInterpreter','none');
addpath(genpath('./core/'));
addpath(genpath('../../main/'));
if use_gpu, Setup(); else, Setup_CPU(false); end
parallel.gpu.rng(0, 'Philox4x32-10');

% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));


%% load

dirs = dir('working/ivy*');
for exp_i = 1:length(dirs)
    
    disp(['### ' num2str(exp_i) '/' num2str(length(dirs)) ' ###']);

    net_id = dirs(exp_i).name;
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
    if use_gpu
        config.gen_net = vl_simplenn_move(loaded_nets.net2, 'gpu');
    else
        config.gen_net = loaded_nets.net2;
    end

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

    % batch
    images_num = 1;
    for i = 1:images_num:size(images_train,4)

        for_time = tic;
        
        % batch
        i_begin = i;
        i_end = min(i+images_num-1, size(images_train,4));
        i_size = i_end - i_begin + 1;

        % images
        syn_mat = images_train(:,:,:,i_begin:i_end);
        gen_im = floor((syn_mat+1)*128);

        % langevin
        if use_gpu
            z = gpuArray(randn([config.z_sz i_size],'single'));
        else
            z = randn([config.z_sz i_size],'single');
        end
        t_max = 1000;
        dists = zeros(t_max, i_size);
        z_min = zeros([config.z_sz i_size]);
        dist_min = inf(i_size);
        for t = 1:t_max
            if use_gpu
                res = vl_gan(config.gen_net, z, syn_mat, [], 'conserveMemory', 1);
            else
                res = vl_gan_cpu(config.gen_net, z, syn_mat, [], 'conserveMemory', 1);
            end
            delta_log = res(1).dzdx / config.refsig2 / config.refsig2 - z;
            z = z + config.Delta2 * config.Delta2 / 2 * delta_log;
            %z = z + config.Delta2 * randn(size(z2), 'single');

            if use_gpu
                gen_im_z = gather(floor((res(end).x+1)*128));
            else
                gen_im_z = floor((res(end).x+1)*128);
            end

            z_cpu = gather(z);
            for j = 1:i_size
                gen_im_j = gen_im(:,:,:,j);
                gen_im_z_j = gen_im_z(:,:,:,j);
                dists(t,j) = sqrt(sum((gen_im_j(:)-gen_im_z_j(:)).^2));
                
                if dists(t,j) < dist_min(j)
                    z_min(:,:,j) = z_cpu(:,:,j);
                    dist_min(j) = dists(t,j);
                end
            end
            
        %     if mod(t, 100) == 0
        %         fprintf('%1.4f\n', dists(t,1));
        %     end
        end

        [images_min_dist(i_begin:i_end), images_min_dist_index(i_begin:i_end)] = min(dists);
        images_zs(:,:,:,i_begin:i_end) = z_min;

        for j = i_begin:i_end
            fprintf('%4.0f / %4.0f -> %4.4f %4.0f (%3.2fs)\n', j, size(images_train,4), images_min_dist(j), images_min_dist_index(j), toc(for_time)/i_size);
        end
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