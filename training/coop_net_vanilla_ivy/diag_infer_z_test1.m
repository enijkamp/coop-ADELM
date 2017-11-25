function [] = diag_infer_z()

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

%% inference

% malformed
% z = randn(config.z_sz,'single');
% syn_mat = vl_gan_cpu(config.gen_net,z);
% syn_mat = syn_mat(end).x;
% syn_mat = zeros(size(syn_mat));
% gen_im = floor((syn_mat+1)*128);

z = randn(config.z_sz,'single');
syn_mat = vl_gan_cpu(config.gen_net,z);
syn_mat = syn_mat(end).x;
gen_im = floor((syn_mat+1)*128);

z2 = randn(config.z_sz,'single');
syn_mat2 = vl_gan_cpu(config.gen_net,z2);
syn_mat2 = syn_mat2(end).x;
gen_im2 = floor((syn_mat2+1)*128);

% langevin
config_train.Delta2 = 0.05; % 0.01;
t_max = 1000;
dists = zeros(t_max, 2);
for t = 1:t_max
    res = vl_gan_cpu(config.gen_net, z2, syn_mat, [], 'conserveMemory', 1);
    delta_log = res(1).dzdx / config_train.refsig2 / config_train.refsig2 - z2;
    z2 = z2 + config_train.Delta2 * config_train.Delta2 / 2 * delta_log;
    %z2 = z2 + config_train.Delta2 * randn(size(z2), 'single');

    gen_im_z2 = floor((res(end).x+1)*128);
    dists(t,1) = sqrt(sum((z(:)-z2(:)).^2));
    dists(t,2) = sqrt(sum((gen_im(:)-gen_im_z2(:)).^2));
    
    if mod(t, 100) == 0
        fprintf('%1.4f %1.4f\n', dists(t,1), dists(t,2));
    end
    
    clear res;
end

% plot
figure;
plot(dists);

syn_mat = vl_gan_cpu(config.gen_net,z);
syn_mat = syn_mat(end).x;
gen_im3 = floor((syn_mat+1)*128);

syn_mat2 = vl_gan_cpu(config.gen_net,z2);
syn_mat2 = syn_mat2(end).x;
gen_im4 = floor((syn_mat2+1)*128);

f1 = figure();
imshow(uint8(gen_im), []);
set(gcf, 'Position', [100, 100, 512, 512])
f2 = figure();
imshow(uint8(gen_im2), []);
set(gcf, 'Position', [100, 100, 512, 512])
f3 = figure();
imshow(uint8(gen_im3), []);
set(gcf, 'Position', [100, 100, 512, 512])
f4 = figure();
imshow(uint8(gen_im4), []);
set(gcf, 'Position', [100, 100, 512, 512])

disp([z(:) z2(:)]);
disp(num2str(sqrt(sum((gen_im(:)-gen_im2(:)).^2))));
disp(num2str(sqrt(sum((gen_im(:)-gen_im4(:)).^2))));

disp('done');

end