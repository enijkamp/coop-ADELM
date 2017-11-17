function [] = experiment_learn_escher_22()

% (22): (19) but keep training des
%
% result:
%
% Run  Delta1   Gamma1   Gamma2   Result             Losss
%   1   0.002     0.07   0.0003   Ok
%   2   0.002     0.07   0.0009   Ok
%   3   0.002     0.07   0.0001   Ok
%   4   0.002     0.04   0.0003   Ok 
%   5   0.002     0.04   0.0009   Ok
%   6   0.002     0.04   0.0001   Ok
%   7   0.002     0.01   0.0003   Fail, uniform      Two peaks in the beginning
%   8   0.002     0.01   0.0009   Fail, blobs        Flat line
%   9   0.002     0.01   0.0001   Fail, rects        Peaks in the beginning
%  10   0.005     0.07   0.0003   Fail, cow blobs    Increases
%  11   0.005     0.07   0.0009   Ok
%  12   0.005     0.07   0.0001   Ok
%  13   0.005     0.04   0.0003   Ok
%  14   0.005     0.04   0.0009   Ok
%  15   0.005     0.04   0.0001   Fail, fuzzy        Looks fine
%  16   0.005     0.01   0.0003   Fail, blobs        Peaks in the beginning
%  17   0.005     0.01   0.0009   ?
%  18   0.005     0.01   0.0001   ?

exp_id = 22;

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
Deltas1 = [0.002, 0.005, 0.009];
Gammas1 = [0.07, 0.04, 0.01];
Gammas2 = [0.0003, 0.0009, 0.0001];

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
config.nIteration = 200;
config.nTileRow = 16;
config.nTileCol = 16;
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
% (1) cooking: fast decay
lr_1_to_50 = logspace(-2, -4, 100)*100;
lr_1_to_50 = lr_1_to_50(1:50);
% (2) refinement 1: slow decay
lr_51_to_150 = logspace(log10(lr_1_to_50(end)), log10(lr_1_to_50(end)/100), 100);
% (3) refinement 2: const
lr_151_to_200 = linspace(lr_51_to_150(end), lr_51_to_150(end), 50);
config.learningRate1 = [lr_1_to_50 lr_51_to_150 lr_151_to_200];

%% net 2 (generator) decay
% (1) cooking: fast decay
lr_1_to_50 = logspace(-2, -4, 100)*100;
lr_1_to_50 = lr_1_to_50(1:50);
% (2) refinement 1: slow decay
lr_51_to_150 = logspace(log10(lr_1_to_50(end)), log10(lr_1_to_50(end)/2), 100);
% (3) refinement 2: const
lr_151_to_200 = linspace(lr_51_to_150(end), lr_51_to_150(end), 50);
config.learningRate2 = [lr_1_to_50 lr_51_to_150 lr_151_to_200];

%% gammas
config.Gammas1 = config.Gamma1 * config.learningRate1;
config.Gammas2 = config.Gamma2 * config.learningRate2; 

% plot
h1 = figure; plot(config.Gammas1); legend('gamma1 (net1 / des)','Location','northeast');
h2 = figure; plot(config.Gammas2); legend('gamma2 (net2 / gen)','Location','northeast');

saveas(h1, [config.working_folder, '/gamma1.fig']);
saveas(h1, [config.working_folder, '/gamma1.png']);
saveas(h2, [config.working_folder, '/gamma2.fig']);
saveas(h2, [config.working_folder, '/gamma2.png']);

save([config.working_folder, '/config.mat'], 'config');

% interpolation
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;

%% run
learn_dualNets_config('escher', exp_type, config);

end
end
end
