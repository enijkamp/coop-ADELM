function [] = exp_texture_joint_12()

% export CPATH=/home/enijkamp/cudnn-3.0/include/
% export LD_LIBRARY_PATH=/home/enijkamp/cudnn-3.0/lib64:$LD_LIBRARY_PATH
% export LIBRARY_PATH=/home/enijkamp/cudnn-3.0/lib64:$LIBRARY_PATH

% sudo CPATH=/home/enijkamp/cudnn-3.0/include/ LD_LIBRARY_PATH=/home/enijkamp/cudnn-3.0/lib64:$LD_LIBRARY_PATH LIBRARY_PATH=/home/enijkamp/cudnn-3.0/lib64:$LIBRARY_PATH /usr/local/MATLAB/R2017a/bin/matlab

exp_num = 12;

% (9) benchmark - increase T=20 (comet k80 no cudnn)

% descriptor
Gamma = 0.00005;
Deltas = 0.3;
Ts = 20;
% generator
Delta2 = 0.3;
Gammas2 = 0.0002; %[0.0002 0.0001 0.00005];

% config
img_name = 'ivy';
img_size = 'all';
patch_size = 64;

% setup
rng(123);
use_gpu = 1;
compile_convnet = 1;
compile_cudnn = 0;

% setup
setup_path();
setup_convnet(use_gpu, compile_convnet, compile_cudnn);
disp(gpuDevice());

% load nets
net1_init = load('../nets/ivy/all/des_net', 'net1');
net2_init = load('../nets/ivy/all/gen_net', 'net2');

% run
num = 1;
total = length(Deltas) * length(Gammas2) * length(Ts);
for Delta = Deltas
    for Gamma2 = Gammas2
        for T = Ts
            % log
            disp(['############################# run ' num2str(num) ' / ' num2str(total) ' #############################']);
            
            % prep
            prefix = [img_name '/' num2str(img_size) '_' num2str(exp_num) '_' num2str(num) '/'];
            [config, net1] = train_coop_config();
            config = prep_images(config, ['../data/' img_name '/' num2str(img_size) '/'], patch_size);
            config = prep_dirs(config, prefix);
            config.use_gpu = use_gpu;
            
            % config
            config.nIteration = 400;
            config.batchSize = 50;

            % sampling parameters
            config.num_syn = 120;
            
            % descriptor net1 parameters
            config.Delta = Delta;
            config.Gamma = [1.0*ones(1,100), 0.5*ones(1,100), 0.1*ones(1,100), 0.05*ones(1,100)] * Gamma;
            config.refsig = 1;                                            
            config.T = T;            
            
            % generator net2 parameters
            config.Delta2 = Delta2;
            config.Gamma2 = [1.0*ones(1,100), 0.5*ones(1,100), 0.1*ones(1,100), 0.05*ones(1,100)] * Gamma2;
            config.refsig2 = 1;
            config.s = 0.3;
            config.real_ref = 1;
            config.cap2 = 8;
            
            % learn
            learn_dual_net(config, net1_init.net1, net2_init.net2);

            num = num + 1;
        end
    end
end
end

function [config] = prep_images(config, patch_path, patch_size)
[mean_im, imdb] = load_images(patch_path, patch_size);
config.mean_im = mean_im;
config.imdb = imdb;
end

function [config] = prep_dirs(config, prefix)
config.trained_folder = [config.trained_folder prefix];
config.gen_im_folder = [config.gen_im_folder prefix];
config.syn_im_folder = [config.syn_im_folder prefix];
if ~exist(config.trained_folder,'dir') mkdir(config.trained_folder); end
if ~exist(config.gen_im_folder,'dir') mkdir(config.gen_im_folder); end
if ~exist(config.syn_im_folder,'dir') mkdir(config.syn_im_folder); end
end

function [mean_im, imdb] = load_images(img_path, img_size)
files = dir([img_path, '*.png']);
imdb = zeros(img_size, img_size,3,length(files));
for i = 1:length(files)
    imdb(:,:,:,i) = imread([img_path,files(i).name]);
end
mean_im = single(sum(imdb,4)/size(imdb,4));
imdb = single(imdb - repmat(mean_im,1,1,1,size(imdb,4)));
end
