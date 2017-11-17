function [] = experiment_learn_escher_16()

% (16): decrease batch size
%
% result: fail - explodes

exp_id = 16;

%% prep
restoredefaultpath();
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

%% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
%assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

%% config
exp_type = 'object';
config = frame_config('escher', 'em', exp_type, num2str(exp_id));

%% override
config.nIteration = 400;
config.nTileRow = 16;
config.nTileCol = 16;
config.batch_size = 50;

% parameters for net 1 (descriptor)
config.T = 10;
config.Delta1 = 0.002; % 0.005
config.Gamma1 = 0.07;
config.refsig1 = 0.016;
config.cap1 = 8;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gamma2 = 0.0003;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;


%% net 1 (descriptor) decay
% cooking: fast decay from (4)
nIteration = 200;
rate_list = logspace(-2, -4, 80)*100;
lr_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
lr_array = reshape(lr_array, 1, []);
lr_1_to_50 = lr_array(1:50);

% refinement: slow decay from (10)
nIteration = 200;
rate_list = logspace(-2.75, -9, 80)*100;
lr_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
lr_array = reshape(lr_array, 1, []);
lr_51_to_200 = lr_array(1:150);

% const: very low gamma
lr_200_to_400 = ones(1,200) * lr_51_to_200(end);

config.learningRate1 = [lr_1_to_50 lr_51_to_200 lr_200_to_400];

%% net 2 (generator) decay
% cooking: fast decay from (4)
nIteration = 200;
rate_list = logspace(-2, -4, 80)*100;
lr_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
lr_array = reshape(lr_array, 1, []);
lr_1_to_50 = lr_array(1:50);

% refinement 1: slow decay
lr_51_to_100 = linspace(0.245, 0.15, 50);

% refinement 2: very slow decay
lr_100_to_400 = linspace(0.15, 0.05, 300);

config.learningRate2 = [lr_1_to_50 lr_51_to_100 lr_100_to_400];


%% gammas
config.Gammas1 = config.Gamma1 * config.learningRate1;
config.Gammas2 = config.Gamma2 * config.learningRate2; 

% plot
h1 = figure; 
plot(config.Gammas1);
legend('gamma1 (net1 / des)','Location','northeast');
h2 = figure; 
plot(config.Gammas2);
legend('gamma2 (net2 / gen)','Location','northeast');

saveas(h1, [config.working_folder, '/gamma1.fig']);
saveas(h1, [config.working_folder, '/gamma1.png']);
saveas(h2, [config.working_folder, '/gamma2.fig']);
saveas(h2, [config.working_folder, '/gamma2.png']);

% interpolation
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;

%% run
learn_dualNets_config('escher', exp_type, config);
