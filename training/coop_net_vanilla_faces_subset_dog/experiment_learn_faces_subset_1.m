function [] = experiment_learn_faces_subset_1()

%% run
%
% sudo CPATH=/home/enijkamp/cudnn-3.0/include/ LD_LIBRARY_PATH=/home/enijkamp/cudnn-3.0/lib64:$LD_LIBRARY_PATH LIBRARY_PATH=/home/enijkamp/cudnn-7.0/lib64:$LIBRARY_PATH /usr/local/MATLAB/R2017a/bin/matlab -softwareopengl
%
% sudo CPATH=/home/paperspace/cudnn-3.0/include/ LD_LIBRARY_PATH=/home/paperspace/cudnn-3.0/lib64:$LD_LIBRARY_PATH LIBRARY_PATH=/home/paperspace/cudnn-3.0/lib64:$LIBRARY_PATH /usr/local/MATLAB/R2017a/bin/matlab -softwareopengl

exp_id = 1;

%% prep
restoredefaultpath();
rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

%% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
assert(strcmp(dev.Name, 'Quadro P5000'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

%% config
num_imgs = 97;
exp_type = 'object';
config = frame_config('faces_subset_dog', 'em', exp_type, num_imgs, num2str(exp_id));

%% run
learn_dualNets_config('faces_subset_dog', exp_type, config, num_imgs);
