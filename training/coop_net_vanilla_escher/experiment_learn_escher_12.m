function [] = experiment_learn_escher_12()

% (12): (1) keep decay for descriptor (fix), increase for generator

exp_id = 12;

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

%% config
exp_type = 'object';
config = frame_config('escher', 'em', exp_type, num2str(exp_id));

%% override
config.nIteration = 200;
config.nTileRow = 12;
config.nTileCol = 12;
config.batch_size = 100;

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
% cooking: slow decay from (4)
nIteration = 200;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
learningRate_array = reshape(learningRate_array, 1, []);
learningRate_1_to_50 = learningRate_array(1:50);

% refinement: fast decay from (10)
nIteration = 200;
rate_list = logspace(-2.75, -9, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
learningRate_array = reshape(learningRate_array, 1, []);
learningRate_51_to_200 = learningRate_array(1:50);

config.learningRate1 = [learningRate_1_to_50 learningRate_51_to_200];

%% net 2 (generator) decay
% cooking: slow decay from (4)
nIteration = 200;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
learningRate_array = reshape(learningRate_array, 1, []);
learningRate_1_to_50 = learningRate_array(1:50);

% refinement: fast decay from (10)
learningRate_51_to_200 = linspace(0.245, 0.01, 150);

config.learningRate2 = [learningRate_1_to_50 learningRate_51_to_200];

% plot
h = figure; 
hold on;
plot(config.learningRate1);
plot(config.learningRate2);
legend('net1 / des','net2 / gen','Location','northeast');
saveas(h, [config.working_folder, '/decay.fig']);
saveas(h, [config.working_folder, '/decay.png']);

% interpolation
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;

%% run
learn_dualNets_config('escher', exp_type, config);
