function [] = experiment_learn_ivy_all()

% gpuArray(randn(...)) -> 11 minutes?
% gpuArray.randn(...)  -> 11 minutes
% without cuDNN -> 55 minutes

exp_id = 2;

%% config
category = ['ivy_' num2str(exp_id)];
exp_type = 'object';
config = frame_config(category, 'em', exp_type);

config.nIteration = 100;
config.nTileRow = 6;
config.nTileCol = 6;

% parameters for net 1
config.T = 10;
config.Delta1 = 0.005;
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

% ?
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;


% gpu
parallel.gpu.rng(0, 'Philox4x32-10');

%% run
learn_dualNets_config(category, exp_type, config);
