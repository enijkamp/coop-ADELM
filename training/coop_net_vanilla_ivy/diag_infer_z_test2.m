function [] = diag_infer_z_test2()

% langevin to sample images

%% prep

% clear
clear all;
close all;
restoredefaultpath();

% prep
set(0,'DefaultTextInterpreter','none');
addpath(genpath('./core/'));
addpath(genpath('../../main/'));
Setup_CPU(false);
rng(123);

%% load

loaded_nets = load('working/ivy_dense_em_11_11/layer_01_iter_400_model.mat');
loaded_config = load('working/ivy_dense_em_11_11/config.mat');
config_train = loaded_config.config;

%% config

% nets
config.des_net = loaded_nets.net1;
config.gen_net = loaded_nets.net2;

% image and latent space
config.z_sz = [1,1,30];
config.im_sz = [64,64,3];
config.mean_im = single(zeros(config.im_sz));


%% inference (1) - malformed image -> 6300 not reachable

% malformed
z = randn(config.z_sz,'single');
syn_mat = vl_gan_cpu(config.gen_net,z);
syn_mat = syn_mat(end).x;
syn_mat = zeros(size(syn_mat)); % set zero
gen_im = floor((syn_mat+1)*128);

z = randn(config.z_sz,'single');
syn_mat2 = vl_gan_cpu(config.gen_net,z);
syn_mat2 = syn_mat2(end).x;
gen_im2 = floor((syn_mat2+1)*128);

% langevin
config_train.Delta2 = 0.05; % override
t_max = 1000;
dists = zeros(t_max, 1);
for t = 1:t_max
    res = vl_gan_cpu(config.gen_net, z, syn_mat, [], 'conserveMemory', 1);
    delta_log = res(1).dzdx / config_train.refsig2 / config_train.refsig2 - z;
    z = z + config_train.Delta2 * config_train.Delta2 / 2 * delta_log;
    %z = z + config_train.Delta2 * randn(size(z2), 'single');

    gen_im_z = floor((res(end).x+1)*128);
    dists(t,1) = sqrt(sum((gen_im(:)-gen_im_z(:)).^2));
    
    if mod(t, 100) == 0
        fprintf('%1.4f\n', dists(t,1));
    end
    
    clear res;
end

% plot
figure;
plot(dists);

syn_mat = vl_gan_cpu(config.gen_net,z);
syn_mat = syn_mat(end).x;
gen_im3 = floor((syn_mat+1)*128);

figure;
hold on;
subplot(1,3,1);
imshow(uint8(gen_im), []);
subplot(1,3,2);
imshow(uint8(gen_im2), []);
subplot(1,3,3);
imshow(uint8(gen_im3), []);
set(gcf, 'Position', [100, 100, 3*256, 256])
hold off;

disp('# result 1 #');
disp(num2str(sqrt(sum((gen_im(:)-gen_im2(:)).^2))));
disp(num2str(sqrt(sum((gen_im(:)-gen_im3(:)).^2))));
disp(' ');


%% inference (2) - synthesis image -> 1400 reachable

%images_train = read_images('../data/ivy/all/', dir('../data/ivy/all/*.png'), [64, 64, 3]);
%syn_mat = images_train(:,:,:,1665);

images_train = read_images('figure/ivy_dense_em_11_11/', dir('figure/ivy_dense_em_11_11/*.png'), [64, 64, 3]);
syn_mat = images_train(:,:,:,1);

gen_im = floor((syn_mat+1)*128);

z = randn(config.z_sz,'single');
syn_mat2 = vl_gan_cpu(config.gen_net,z);
syn_mat2 = syn_mat2(end).x;
gen_im2 = floor((syn_mat2+1)*128);

% langevin
config_train.Delta2 = 0.05;
t_max = 1000;
dists = zeros(t_max, 1);
for t = 1:t_max
    res = vl_gan_cpu(config.gen_net, z, syn_mat, [], 'conserveMemory', 1);
    delta_log = res(1).dzdx / config_train.refsig2 / config_train.refsig2 - z;
    z = z + config_train.Delta2 * config_train.Delta2 / 2 * delta_log;
    %z = z + config_train.Delta2 * randn(size(z2), 'single');

    gen_im_z = floor((res(end).x+1)*128);
    dists(t,1) = sqrt(sum((gen_im(:)-gen_im_z(:)).^2));
    
    if mod(t, 100) == 0
        fprintf('%1.4f\n', dists(t,1));
    end
    
    clear res;
end

% plot
figure;
plot(dists);

syn_mat = vl_gan_cpu(config.gen_net,z);
syn_mat = syn_mat(end).x;
gen_im3 = floor((syn_mat+1)*128);

figure;
hold on;
subplot(1,3,1);
imshow(uint8(gen_im), []);
subplot(1,3,2);
imshow(uint8(gen_im2), []);
subplot(1,3,3);
imshow(uint8(gen_im3), []);
set(gcf, 'Position', [100, 100, 3*256, 256])
hold off;

disp('# result 2 #');
disp(num2str(sqrt(sum((gen_im(:)-gen_im2(:)).^2))));
disp(num2str(sqrt(sum((gen_im(:)-gen_im3(:)).^2))));
disp(' ');


%% inference (3) - training image -> 5100 somewhat reachable

images_train = read_images('../data/ivy/all/', dir('../data/ivy/all/*.png'), [64, 64, 3]);
syn_mat = images_train(:,:,:,1665);

gen_im = floor((syn_mat+1)*128);

z = randn(config.z_sz,'single');
syn_mat2 = vl_gan_cpu(config.gen_net,z);
syn_mat2 = syn_mat2(end).x;
gen_im2 = floor((syn_mat2+1)*128);

% langevin
config_train.Delta2 = 0.05;
t_max = 1000;
dists = zeros(t_max, 1);
for t = 1:t_max
    res = vl_gan_cpu(config.gen_net, z, syn_mat, [], 'conserveMemory', 1);
    delta_log = res(1).dzdx / config_train.refsig2 / config_train.refsig2 - z;
    z = z + config_train.Delta2 * config_train.Delta2 / 2 * delta_log;
    %z = z + config_train.Delta2 * randn(size(z2), 'single');

    gen_im_z = floor((res(end).x+1)*128);
    dists(t,1) = sqrt(sum((gen_im(:)-gen_im_z(:)).^2));
    
    if mod(t, 100) == 0
        fprintf('%1.4f\n', dists(t,1));
    end
    
    clear res;
end

% plot
figure;
plot(dists);

syn_mat = vl_gan_cpu(config.gen_net,z);
syn_mat = syn_mat(end).x;
gen_im3 = floor((syn_mat+1)*128);

figure;
hold on;
subplot(1,3,1);
imshow(uint8(gen_im), []);
subplot(1,3,2);
imshow(uint8(gen_im2), []);
subplot(1,3,3);
imshow(uint8(gen_im3), []);
set(gcf, 'Position', [100, 100, 3*256, 256])
hold off;

disp('# result 3 #');
disp(num2str(sqrt(sum((gen_im(:)-gen_im2(:)).^2))));
disp(num2str(sqrt(sum((gen_im(:)-gen_im3(:)).^2))));
disp(' ');


disp('done');

end

function [img_mat] = read_images(inPath, files, imageSize)

if isempty(files)
    fprintf(['error: No training images are found in "' inPath '"\n']);
    keyboard;
end

img_mat = zeros([imageSize, length(files)], 'single');
for iImg = 1:length(files)
    img = single(imread(fullfile(inPath, files(iImg).name)));
    img = imresize(img, imageSize(1:2));
    min_val = min(img(:));
    max_val = max(img(:));
    img_mat(:,:,:,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
end

end