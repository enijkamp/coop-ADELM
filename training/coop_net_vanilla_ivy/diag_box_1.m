function [] = diag_box_1()

% clear
clear all;
close all;
restoredefaultpath();

% plot energy over samples
exp_id = 1;

% prep
set(0,'DefaultTextInterpreter','none');
addpath(genpath('./core/'));
Setup_CPU(false);
rng(123);

% malformed
neg = ["em_1_1", "em_2_1", "em_3_3", "em_3_4", "em_4_1", "em_6_8", "em_10_1",...
    "em_10_2", "em_10_5", "em_10_6", "em_11_2", "em_11_6"];


%% run 1

% load
load('working/ivy_dense_em_0_1/layer_01_iter_2200_model.mat'); % descriptor
load('working/ivy_dense_em_0_1/config.mat');

% plot
out_dir = ['diag/box_' num2str(exp_id) '_1/'];
plot_box_energy(out_dir, neg, config, net1, false);


%% run 2

% load
load('working/ivy_dense_em_0_2/layer_01_iter_1200_model.mat'); % descriptor
load('working/ivy_dense_em_0_2/config.mat');

% plot
out_dir = ['diag/box_' num2str(exp_id) '_2/'];
plot_box_energy(out_dir, neg, config, net1, false);


%% run 3

% load
load('../multigrid_code/working/ivy_1_1/iter_400_model.mat'); % descriptor
load('../multigrid_code/working/ivy_1_1/config.mat');

% plot
out_dir = ['diag/box_' num2str(exp_id) '_3/'];
plot_box_energy(out_dir, neg, config, net3, true);


%% run 4

% load
load('../multigrid_code/working/ivy_4_1/iter_400_model.mat'); % descriptor
load('../multigrid_code/working/ivy_4_1/config.mat');

% plot
out_dir = ['diag/box_' num2str(exp_id) '_4/'];
plot_box_energy(out_dir, neg, config, net3, true);

disp('done');

end