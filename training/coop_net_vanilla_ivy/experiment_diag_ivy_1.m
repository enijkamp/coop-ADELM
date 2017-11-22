function [] = experiment_diag_ivy_1()

%% prep
clear all;
close all;
restoredefaultpath();

set(0,'DefaultTextInterpreter','none');
addpath(genpath('./core/'));
Setup_CPU(false);
rng(123);

%% exp
exp_id = 1;

%% load
load('working/ivy_dense_em_0_1/layer_01_iter_2200_model.mat'); % descriptor
load('working/ivy_dense_em_0_1/config.mat');

% paths
exps = {};

% train
exps{end+1} = { '../data/ivy/all/', '../data/ivy/all/*.png', 'train' };

% old - mean substraction
exps{end+1} = { '../ims_syn/ivy/all_4_1/', '../ims_syn/ivy/all_4_1/*.png', 'all_4_1 1' };
exps{end+1} = { '../ims_gen/ivy/all_4_1/', '../ims_gen/ivy/all_4_1/*.png', 'all_4_1 2' };

% new - ivy
dirs = dir('figure/ivy*');
for i = 1:length(dirs)
    exp_path = [ dirs(i).folder '/' dirs(i).name];
    if ~isempty(dir([exp_path '/net1*.png'])) && ~isempty(dir([exp_path '/net2*.png']))
        name_parts = strsplit(dirs(i).name,'_');
        name = [name_parts{end-1} '_' name_parts{end}];
        exps{end+1} = { exp_path, [exp_path '/net1*.png'], [name ' 1'] };
        exps{end+1} = { exp_path, [exp_path '/net2*.png'], [name ' 2'] };
    end
end

% energy
energies = {};
groups = {};
for i = 1:length(exps)
    exp_path = exps{i};
    images = read_images(exp_path{1}, dir(exp_path{2}), [64, 64, 3]);
    energies{i} = compute_energy(config,net1,images);
    groups = horzcat(groups, repmat({exp_path{3}},1,length(energies{i})));
end

% box
f1 = figure('pos',[10 10 2000 1000]);
hold on;
boxplot(flat(energies), groups);
xtickangle(90);
set(gca,'fontsize',8);
set(gca, 'LooseInset', get(gca,'TightInset'));
hold off;

% save
mkdir(['energy/experiment_diag_ivy_' num2str(exp_id)]);
save(['energy/experiment_diag_ivy_' num2str(exp_id) '/energy.mat'], 'energies', 'groups');
saveas(f1, ['energy/experiment_diag_ivy_' num2str(exp_id) '/energy_box.fig']);
saveas(f1, ['energy/experiment_diag_ivy_' num2str(exp_id) '/energy_box.png']);
saveas(f1, ['energy/experiment_diag_ivy_' num2str(exp_id) '/energy_box.pdf']);

disp('done');

end

function out = flat(energies)
out = [];
for i = 1:length(energies)
    en = energies{i};
    out = horzcat(out, en');
end
end

function en = compute_energy(config,net1,images)
en = zeros(size(images,4), 1);
for i = 1:size(images,4)
    en(i) = get_im_energy(config,net1,images(:,:,:,i));
end
end

function [img_mat] = read_images(inPath, files, imageSize)

if isempty(files)
    fprintf(['error: No training images are found in "' inPath '"\n']);
    keyboard;
end

img_mat = zeros([imageSize, length(files)], 'single');
for iImg = 1:length(files)
    fprintf('read and process images %d / %d\n', iImg, length(files))
    img = single(imread(fullfile(inPath, files(iImg).name)));
    img = imresize(img, imageSize(1:2));
    min_val = min(img(:));
    max_val = max(img(:));
    img_mat(:,:,:,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
end

end