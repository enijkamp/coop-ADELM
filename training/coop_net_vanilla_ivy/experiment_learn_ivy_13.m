function [] = experiment_learn_ivy_13()

clear all;
close all;
restoredefaultpath();

%% exp

exp_id = 13;

% random mean, search Gamma1
%
% result: Gamma1=0.004
%

%% prep
diary(['experiment_learn_ivy_' num2str(exp_id) '.out']);
Setup(true, false);

%% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

%% params

% mitch
% Deltas1 = [0.15];
% Gammas1 = [0.002];
% Gammas2 = [0.0002];

% (13)
Deltas1 = [0.15];
Gammas1 = [0.002 0.004 0.008 0.016 0.032];
Gammas2 = [0.0002];

num = 0;
for Delta1 = Deltas1
for Gamma1 = Gammas1
for Gamma2 = Gammas2
    
%% seed
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

%% output
num = num + 1;
disp(['### ' num2str(num) '/' num2str(length(Deltas1)*length(Gammas1)*length(Gammas2)) ' ###']);

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
config.Delta1 = Delta1;
config.Gammas1 = (logspace(-2, -3, 200)*100) * Gamma1;
config.refsig1 = 1;
config.cap1 = 20;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gammas2 = (logspace(-2, -4, 200)*100) * Gamma2;
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
