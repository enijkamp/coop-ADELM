function [] = experiment_learn_ivy_31()

%% reset
clear all;
close all;
restoredefaultpath();

%% exp

exp_id = 31;

% random mean, infer z
%
% result: ?
%

%% prep
diary(['experiment_learn_ivy_' num2str(exp_id) '.out']);
Setup(true, false);

%% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
%assert(strcmp(dev.Name, 'Quadro P5000'));
%assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(strcmp(dev.Name, 'Tesla P100-PCIE-16GB'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

%% params

% mitch
% Deltas1 = [0.15];
% Gammas1 = [0.002];
% Gammas2 = [0.0002];

% (13)
% Deltas1 = [0.15];
% Gammas1 = [0.002 0.004 0.008 0.016 0.032];
% Gammas2 = [0.0002];

% (14)
% Deltas1 = [0.15];
% Gammas1 = [0.004];
% Gammas2 = [0.0001 0.0002 0.0004 0.0008 0.002 0.004];

% (15)
% Deltas1 = [0.15];
% Gammas1 = [0.004];
% Gammas2 = [0.0008];
% Decay1 = logspace(-2, -3, 200)*100;
% Decay2 = logspace(-2, -3, 200)*100;

% (16) Mitch, not sharp synthesis
% Deltas1 = [0.15];
% Gammas1 = [0.002];
% Gammas2 = [0.0002];
% Decay1 = ones(1,100); %logspace(-2, -3, 200)*100;
% Decay2 = ones(1,100); %logspace(-2, -3, 200)*100;

% (17) Mitch, bug - used randn() instead of rand() for random mean -> good, generator could be better
% Deltas1 = [0.15];
% Gammas1 = [0.002];
% Gammas2 = [0.0002];
% Decay1 = ones(1,100); %logspace(-2, -3, 200)*100;
% Decay2 = ones(1,100); %logspace(-2, -3, 200)*100;

% (18) linear decay, epochs=200
% Deltas1 = [0.15];
% Gammas1 = [0.002];
% Gammas2 = [0.0002];
% Decay1 = linspace(1,0.1,200); %logspace(-2, -3, 200)*100;
% Decay2 = linspace(1,0.1,200); %logspace(-2, -3, 200)*100;

% (30) infer z
Deltas1 = [0.15];
%Deltas2 = [0.001 0.01 0.1];
Deltas2 = [0.01];
Gammas1 = [0.002];
Gammas2 = [0.0002 0.002 0.02];
Decay1 = linspace(1,0.1,200);
Decay2 = linspace(1,0.1,200);
infer_z = true;

num = 0;
for Delta1 = Deltas1
for Delta2 = Deltas2
for Gamma1 = Gammas1
for Gamma2 = Gammas2
    
%% seed
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

%% output
num = num + 1;
disp(['### ' num2str(num) '/' num2str(length(Deltas1)*length(Deltas2)*length(Gammas1)*length(Gammas2)) ' ###']);

%% config
exp_type = 'object';
config = frame_config('ivy', 'em', exp_type, [num2str(exp_id) '_' num2str(num)]);

%% override
config.nIteration = 200;
config.batch_size = 50;
config.normalize_images = false;
config.random_mean = true;
config.infer_z = infer_z;

% sampling parameters
config.nTileRow = 8;
config.nTileCol = 8;

% parameters for net 1 (descriptor)
config.T = 15;
config.Delta1 = Delta1;
config.Gammas1 = Decay1 * Gamma1;
config.refsig1 = 1;
config.cap1 = 20;

% parameters for net 2 (generator)
config.Delta2 = Delta2;
config.Gammas2 = Decay2 * Gamma2;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;

% plot
h1 = figure; plot(config.Gammas1);
legend('gamma1 (net1 / des)','Location','northeast');
h2 = figure; plot(config.Gammas2);
legend('gamma2 (net2 / gen)','Location','northeast');

saveas(h1, [config.working_folder, '/gamma1.fig']);
saveas(h1, [config.working_folder, '/gamma1.png']);
saveas(h2, [config.working_folder, '/gamma2.fig']);
saveas(h2, [config.working_folder, '/gamma2.png']);

% interpolation
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;

% config
save([config.working_folder, '/config.mat'], 'config');

%% run
learn_dualNets_config('ivy', exp_type, config);

end
end
end
end
