function [] = experiment_compare_code()

clear all;
close all;
restoredefaultpath();

%% exp

exp_id = 99;

% differences:
% - frame_gan_params(): Mitch code does not insert bnorm operator
% - accumulate_gradients2(): Mitch code accumulate_gradients2 gradient not divided by batch size
% - accumlate_gradients1(): "res_l = min(l+1, length(res));" -> "res_l = min(l+2, length(res));"
%
% result: ?
%

%% prep
Setup(0, 0);
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

%% params
num = 0;

%% output
num = num + 1;

%% config
exp_type = 'object';
config = frame_config('ivy', 'em', exp_type, [num2str(exp_id) '_' num2str(num)]);

%% override
config.nIteration = 4;
config.nTileRow = 8;
config.nTileCol = 8;
config.batch_size = 50;
config.normalize_images = false;
config.subtract_mean = true;

% parameters for net 1 (descriptor)
config.T = 10;
config.Delta1 = 0.3;
config.Gammas1 = ones(1,100) * 0.07;
config.refsig1 = 0.016;
config.cap1 = 20;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gammas2 = ones(1,100) * 0.0003;
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
