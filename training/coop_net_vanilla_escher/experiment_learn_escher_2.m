function [] = experiment_learn_escher_2()

% (2): reduced batch size

exp_id = 2;

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
config.nIteration = 1000;
config.nTileRow = 12;
config.nTileCol = 12;
config.batch_size = 40; % 100

% parameters for net 1
config.T = 10;
config.Delta1 = 0.005; %0.002;
config.Gamma1 = 0.07;
config.refsig1 = 0.016;
config.cap1 = 8;

% parameters for net 2
config.Delta2 = 0.3;
config.Gamma2 = 0.0003;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;

% decay
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(config.nIteration / length(rate_list))),1); %logspace(-2, -4, 60) ones(1,60, 'single')
learningRate_array = reshape(learningRate_array, 1, []);
config.learningRate = learningRate_array;

% interpolation
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;

%% run
learn_dualNets_config('escher', exp_type, config);
