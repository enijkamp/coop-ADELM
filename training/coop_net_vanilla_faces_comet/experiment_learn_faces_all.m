function [] = experiment_learn_faces_all()

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
num_imgs = 97;
exp_type = 'object';
config = frame_config('faces_subset', 'em', exp_type, num_imgs);

%% run
learn_dualNets_config('faces_subset', exp_type, config, num_imgs);
