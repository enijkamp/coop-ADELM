function [config,net_cpu] = train_coop_config

    % string for data path
    config.file_str = 'ivy/all/';
    config.process_ims = 0; % 1 to create image patches from original im
    config.process_str = 'ivy/original/ivy.jpg'; %name of original image
    config.num_patch = 500;
    config.resize_factor = exp(linspace(-1.7,1.2,4));
    config.substract_mean = true;
    
    % batch function
    fn = @(imdb,batch)getBatch(imdb,batch);
    config.getBatch = fn;
    
    % num epochs
    config.nIteration = 80;
    
    % parameters for sampling
    config.im_size = 64;
    config.layer_to_learn = 1;
    config.batchSize = 50;

    % sampling parameters
    config.T = 10;
    config.num_syn = 120;
    
    % standard deviation for reference model q(I/sigma^2)
    config.refsig = 1;

    % descriptor
    config.Delta = 0.3; 
    config.Gamma = 0.00005;

    % generator
    config.Delta2 = 0.3;
    config.Gamma2 = 0.00005;
    config.refsig2 = 1;
    config.s = 0.3;
    config.real_ref = 1;
    config.cap2 = 8;
    
    % image path: where the dataset locates
    config.inPath = '../data/';
     
    % model path: where the deep learning model locates
    config.model_path = '../model/';
    config.model_name = 'imagenet-vgg-verydeep-16.mat';
     %set up empty net
    net_cpu = load([config.model_path, config.model_name]);
    net_cpu = net_cpu.net;

    % name folders for results
    config.syn_im_folder = '../ims_syn/';
    config.gen_im_folder = '../ims_gen/';
    config.trained_folder = '../nets/';
    
end

function im = getBatch(imdb, batch)
    im = imdb(:,:,:,batch) ;
end