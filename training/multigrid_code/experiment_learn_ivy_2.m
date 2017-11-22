function [] = experiment_learn_ivy_2()

clear all;
close all;
restoredefaultpath();

%% exp

exp_id = 2;

% (1) more epochs
%
% result: ?
%

%% prep
diary(['experiment_learn_ivy_' num2str(exp_id) '.out']);
Setup();
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

%% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
%assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(strcmp(dev.Name, 'Tesla P100-PCIE-16GB'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

%% params

% (0) vanilla
Deltas = [0.3];
Gammas = [0.3];

num = 0;
for Delta = Deltas
for Gamma = Gammas

%% output
num = num + 1;
disp(['### ' num2str(num) '/' num2str(length(Deltas)*length(Gammas)) ' ###']);

%% config
config = frame_config(['ivy_' num2str(exp_id) '_' num2str(num)]);
config.inPath = '../data/ivy/all/';
config.force_learn = true;
config.batch_size = 100;

config.numEpochs = 1250;
config.nTileRow = 10;
config.nTileCol = config.nTileRow;

% parameters for net 1
config.T = 30; % 30
config.Delta = Delta;%0.2
config.Gamma = Gamma; % 0.2
config.refsig = 50;%10
config.cap = 1; % 5

% config
save([config.working_folder, '/config.mat'], 'config');

%% run
[net1, net2, net3, config] = learn_multigrid_config(config);

end
end
end