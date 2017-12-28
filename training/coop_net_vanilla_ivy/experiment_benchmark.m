function [] = experiment_benchmark()

%% result: cudnn 5x faster

%% cudnn disabled

% prep
restoredefaultpath();
Setup(false, true);
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

% config
exp_type = 'object';
config = frame_config('ivy', 'em', exp_type, 'benchmark');

% override
config.nIteration = 40;
config.nTileRow = 6;
config.nTileCol = 6;
config.batch_size = 100;

% parameters for net 1 (descriptor)
config.T = 10;
config.Delta1 = 0.005;
config.Gamma1 = 0.07;
config.refsig1 = 0.016;
config.cap1 = 8;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gamma2 = 0.005;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;

% net 1 (descriptor) decay
nIteration = 3000;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
learningRate_array = reshape(learningRate_array, 1, []);

config.learningRate1 = learningRate_array;

% net 2 (generator) decay
nIteration = 3000;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
learningRate_array = reshape(learningRate_array, 1, []);

config.learningRate2 = learningRate_array;

% gammas
config.Gammas1 = config.Gamma1 * config.learningRate1;
config.Gammas2 = config.Gamma2 * config.learningRate2; 

% run
tic_cudnn_0 = tic();
learn_dualNets_config('ivy', exp_type, config);
toc_cudnn_0 = toc(tic_cudnn_0);




%% cudnn enabled

% prep
restoredefaultpath();
Setup(true, true);
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

% config
exp_type = 'object';
config = frame_config('ivy', 'em', exp_type, 'benchmark');

% override
config.nIteration = 40;
config.nTileRow = 6;
config.nTileCol = 6;
config.batch_size = 100;

% parameters for net 1 (descriptor)
config.T = 10;
config.Delta1 = 0.005;
config.Gamma1 = 0.07;
config.refsig1 = 0.016;
config.cap1 = 8;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gamma2 = 0.005;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;

% net 1 (descriptor) decay
nIteration = 3000;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
learningRate_array = reshape(learningRate_array, 1, []);

config.learningRate1 = learningRate_array;

% net 2 (generator) decay
nIteration = 3000;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1);
learningRate_array = reshape(learningRate_array, 1, []);

config.learningRate2 = learningRate_array;

% gammas
config.Gammas1 = config.Gamma1 * config.learningRate1;
config.Gammas2 = config.Gamma2 * config.learningRate2; 


% run
tic_cudnn_1 = tic();
learn_dualNets_config('ivy', exp_type, config);
toc_cudnn_1 = toc(tic_cudnn_1);


%% result
disp(toc_cudnn_0);
disp(toc_cudnn_1);
disp(toc_cudnn_0/toc_cudnn_1);


end
