function [] = test_learn_dual_net()
% simple example of training coop-net on CPU

rng(123);

% config
img_name = 'ivy';
img_size = 'all';
patch_size = 64;

% setup
use_gpu = false;
compile_convnet = false;

% setup convnet
setup_convnet(use_gpu, compile_convnet);

% prep
prefix = ['test/' num2str(img_size) '/'];
[config, net1] = train_coop_config();
config = prep_images(config, ['../data/' img_name '/' num2str(img_size) '/'], patch_size);
config = prep_dirs(config, prefix);

% config
config.use_gpu = use_gpu;
config.nIteration = 10;
config.num_syn = 2;
config.Gamma = ones(1,100) * config.Gamma;
config.Gamma2 = ones(1,100) * config.Gamma2;

% run
learn_dual_net(config, net1);

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