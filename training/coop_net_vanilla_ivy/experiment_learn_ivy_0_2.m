function [] = experiment_learn_ivy_0_2()

% vanilla coop, (0_1) with less epochs
%
% result: good

exp_id = 0;

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
Deltas1 = [0.005];
Gammas1 = [0.07];
Gammas2 = [0.0003];

%num = 0;
num = 1;
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
config.nIteration = 1200; %2200;
config.nTileRow = 6;
config.nTileCol = 6;
config.batch_size = 100;

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
nIteration = 1400; % 3000;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1); %logspace(-2, -4, 60) ones(1,60, 'single')
learningRate_array = reshape(learningRate_array, 1, []);

config.learningRate1 = learningRate_array;

%% net 2 (generator) decay
nIteration = 1400; % 3000;
rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(nIteration / length(rate_list))),1); %logspace(-2, -4, 60) ones(1,60, 'single')
learningRate_array = reshape(learningRate_array, 1, []);

config.learningRate2 = learningRate_array;


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
