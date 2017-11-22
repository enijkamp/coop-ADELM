function [] = experiment_learn_ivy_7()

clear all;
close all;
restoredefaultpath();

%% exp

exp_id = 7;

% continue (6) epochs to 200
%
% result: stop, loss highly fluctuating

%% prep
restoredefaultpath();
diary(['experiment_learn_ivy_' num2str(exp_id) '.out']);
%Setup();
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
%Deltas1 = [0.001 0.002]; % 0.002
%Gammas1 = [0.04 0.08]; % 0.07
%Gammas2 = [0.0001 0.0003 0.0005 0.0008]; % 0.0003

% Deltas1 = [0.002];
% Gammas1 = [0.001 0.005 0.01 0.02 0.05 0.1];
% Gammas2 = [0.0003];

% Deltas1 = [0.002];
% Gammas1 = [0.0001 0.0005 0.001 0.002 0.003 0.004 0.005];
% Gammas2 = [0.0003];

% Deltas1 = [0.001 0.002 0.004];
% Gammas1 = [0.004];
% Gammas2 = [0.0003];

% Deltas1 = [0.004];
% Gammas1 = [0.004];
% Gammas2 = [0.00001 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.001];

Deltas1 = [0.004];
Gammas1 = [0.004];
Gammas2 = [0.0003];

num = 0;
for Delta1 = Deltas1
for Gamma1 = Gammas1
for Gamma2 = Gammas2

%% output
num = num + 1;
disp(['### ' num2str(num) '/' num2str(length(Deltas1)*length(Gammas1)*length(Gammas2)) ' ###']);

%% config
exp_type = 'object';
config = frame_config('ivy', 'em', exp_type, [num2str(exp_id) '_' num2str(num)]);

%% override
config.nIteration = 200;
config.nTileRow = 10;
config.nTileCol = 10;
config.batch_size = 50; %100;

% parameters for net 1 (descriptor)
config.T = 10;
config.Delta1 = Delta1;
config.Gamma1 = Gamma1;
config.refsig1 = 0.016;
config.cap1 = 8;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gamma2 = Gamma2;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;


%% net 1 (descriptor) decay
config.learningRate1 = logspace(-2, -4, config.nIteration)*100;

%% net 2 (generator) decay
config.learningRate2 = logspace(-2, -3, config.nIteration)*100;


%% gammas
config.Gammas1 = config.Gamma1 * config.learningRate1;
config.Gammas2 = config.Gamma2 * config.learningRate2; 

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
