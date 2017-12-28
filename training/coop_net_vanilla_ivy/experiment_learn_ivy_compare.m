function [] = experiment_learn_ivy_compare()

clear all;
close all;
restoredefaultpath();

%% exp

exp_id = 100;

% compare with mitch code
%
% result:
% - used gaussian randn 
% - used wrong refsig1

%% prep
diary(['experiment_learn_ivy_' num2str(exp_id) '.out']);
Setup(true, false);
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

%% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

%% output
num = 1;

%% config
exp_type = 'object';
config = frame_config('ivy', 'em', exp_type, [num2str(exp_id) '_' num2str(num)]);

%% override
config.nIteration = 100;
config.batch_size = 50;
config.normalize_images = false;
config.random_mean = true;

% sampling parameters
config.nTileRow = 8;
config.nTileCol = 8;

% parameters for net 1 (descriptor)
config.T = 15;
config.Delta1 = 0.15;
config.Gammas1 = ones(1, config.nIteration) * 0.002;
config.refsig1 = 1;
config.cap1 = 20;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gammas2 = ones(1, config.nIteration) * 0.0002;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;


% interpolation
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;

% config
save([config.working_folder, '/config.mat'], 'config');

%% run
learn_dualNets_config('ivy', exp_type, config);
