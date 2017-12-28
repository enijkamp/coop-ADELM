function [] = exp_texture_joint_compare()

%% exp
exp_num = 100;

% compare with vanilla code

% image
img_name = 'ivy';
img_size = 'all';

%% setup
rng(123);
use_gpu = 1;
compile_convnet = 0;
compile_cudnn = 1;

setup_path();
setup_convnet(use_gpu, compile_convnet, compile_cudnn);
disp(gpuDevice());

%% run
num = 1;

% config
prefix = [img_name '/' num2str(img_size) '_' num2str(exp_num) '_' num2str(num) '/'];
[config] = train_coop_config();

%% override
config.nIteration = 100;
config.batchSize = 50;
config.substract_mean = false;
config.random_mean = true;

% load
config = prep_dirs(config, prefix);
config = load_images(config);
config.use_gpu = use_gpu;

% sampling parameters
config.num_syn = 64;

% descriptor net1 parameters
config.Delta = 0.15;
config.Gamma = ones(1, config.nIteration) * 0.002;
config.refsig = 1;                                            
config.T = 15;            

% generator net2 parameters
config.Delta2 = 0.3;
config.Gamma2 = ones(1, config.nIteration) * 0.0002;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;

% learn
learn_dual_net(config);

end

function root = setup_path()
root = '../../';
addpath([root 'training/coop_net_code_gpu']);
end

