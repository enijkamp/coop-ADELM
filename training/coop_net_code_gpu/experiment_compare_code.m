function [] = experiment_compare_code()

% compare code with vanilla coop-net
%
% should be numerically equivalent now

restoredefaultpath();

rng(123);

% config
img_name = 'ivy';
img_size = 'all';
patch_size = 64;

% matconvnet
use_gpu = 1;
use_cudnn = 1;
compile_convnet = 0;

setup_convnet(use_gpu, compile_convnet, use_cudnn);

rng(123);
parallel.gpu.rng(0, 'Philox4x32-10');

% prep
prefix = ['99/' num2str(img_size) '/'];
config = train_coop_config();
config = prep_images(config, ['../data/' img_name '/' num2str(img_size) '/'], patch_size);
config = prep_dirs(config, prefix);

%% override
config.use_gpu = use_gpu;
config.nIteration = 4;
config.num_syn = 8*8;
config.batchSize = 50;
config.substract_mean = true;

% parameters for net 1 (descriptor)
config.T = 10;
config.Delta = 0.3;
config.Gamma = ones(1,100) * 0.07;
config.refsig = 0.016;
config.cap1 = 8;

% parameters for net 2 (generator)
config.Delta2 = 0.3;
config.Gamma2 = ones(1,100) * 0.0003;
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;


% run
learn_dual_net(config);

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