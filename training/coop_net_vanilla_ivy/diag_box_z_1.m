function [] = diag_box_z_1()

% clear
clear all;
close all;
restoredefaultpath();

% plot distances
exp_id = 1;

% prep
set(0,'DefaultTextInterpreter','none');
addpath(genpath('./core/'));
Setup_CPU(false);
rng(123);

% malformed
neg = [
    "em_1_1", ...
    "em_2_1", ...
    "em_3_3", ...
    "em_3_4", ...
    "em_4_1", ...
    "em_6_8", ...
    "em_10_1", ...
    "em_10_2", ...
    "em_10_5", ...
    "em_10_6", ...
    "em_11_2", ...
    "em_11_6", ...
    "em_12_6", ...
    "em_12_9", ...
    ];

% exps
dirs_exp = dir('diag/infer_z_1/ivy*');
[~, dirs_ind] = natsortfiles({dirs_exp.name});
dirs = dirs_exp(dirs_ind);

% plot
out_dir = ['diag/box_z_' num2str(exp_id) '_1/'];
plot_box_z(dirs, out_dir, neg);


disp('done');

end