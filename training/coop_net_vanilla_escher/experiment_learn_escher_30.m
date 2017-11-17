function [] = experiment_learn_escher_30()

% (30): (28) lower gamma2 decay
%
% result: ?

exp_id = 30;

%% prep
restoredefaultpath();
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
Deltas1 = [0.001]; % 0.002
Gammas1 = [0.08]; % 0.07
Gammas2 = [0.0005]; % 0.0003

num = 0;
for Delta1 = Deltas1
for Gamma1 = Gammas1
for Gamma2 = Gammas2

%% output
num = num + 1;
disp(['### ' num2str(num) '/' num2str(length(Deltas1)*length(Gammas1)*length(Gammas2)) ' ###']);

%% config
exp_type = 'object';
config = frame_config('escher', 'em', exp_type, [num2str(exp_id) '_' num2str(num)]);

%% override
config.nIteration = 200; %120;
config.nTileRow = 16;
config.nTileCol = 16;
config.batch_size = 100;

% parameters for net 1 (descriptor)
config.T = 10;
config.Delta1 = Delta1; % 0.002
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
% (1) cooking: fast decay
nIteration = 200;
rate_list = logspace(-2, -4, 80)*100;
lr_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
lr_array = reshape(lr_array, 1, []);
lr_1_to_50 = lr_array(1:50);
% (2) refinement: slow decay
nIteration = 200;
rate_list = logspace(log10(lr_1_to_50(end)), -7, 80);
lr_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
lr_array = reshape(lr_array, 1, []);
lr_51_to_200 = lr_array(1:150);
% (3) const: very low gamma
lr_200_to_400 = ones(1,200) * lr_51_to_200(end);

config.learningRate1 = [lr_1_to_50 lr_51_to_200 lr_200_to_400];

%% net 2 (generator) decay
% (1) cooking: fast decay
nIteration = 200;
%rate_list = logspace(-2, -4, 80)*100;
rate_list = logspace(-2, -3, 80)*100;
lr_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
lr_array = reshape(lr_array, 1, []);
lr_1_to_50 = lr_array(1:50);
% (2) refinement 1: slow decay
%lr_51_to_100 = linspace(lr_1_to_50(end), 0.15, 50);
%lr_51_to_100 = linspace(lr_1_to_50(end), 0.2, 50);
lr_51_to_100 = linspace(lr_1_to_50(end), 0.4, 50);
% (3) refinement 2: very slow decay
%lr_100_to_400 = linspace(lr_51_to_100(end), 0.05, 300);
%lr_100_to_400 = linspace(lr_51_to_100(end), 0.1, 300);
%lr_100_to_400 = linspace(lr_51_to_100(end), 0.2, 300);
lr_100_to_400 = linspace(lr_51_to_100(end), 0.4, 300);

config.learningRate2 = [lr_1_to_50 lr_51_to_100 lr_100_to_400];


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
learn_dualNets_config('escher', exp_type, config);

end
end
end
