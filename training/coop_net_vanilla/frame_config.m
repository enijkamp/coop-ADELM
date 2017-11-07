function config = frame_config(category, learn_scheme, exp_type)

% we only support gpu
config.gpus = [1];


if nargin < 1
    category = 'cat';
end

if nargin < 2
   learn_scheme = 'em'; 
end

if nargin < 3
    exp_type = 'texture';
end


% learn scheme can be 1) em, 2) alt
config.learn_scheme = learn_scheme;

% category name
config.categoryName = category;

% model type: can either be dense model or spase model
config.model_type = 'dense';

% experiment type: currently supports 'object', 'texture', 'codebook' and
% 'video'

% image path: where the dataset locates
data_path = '../../data/';
switch category
    case {'celebA', 'face1000'}
        config.inPath = [data_path, category];
        config.datatype = 'celebA';
        config.is_crop = true;
        config.cropped_sz = 108;
    case {'celebB'}
        config.inPath = [data_path, category];
        config.datatype = 'celebB';
        config.is_crop = true;
        config.cropped_sz = 108;
    otherwise
        % own dataset
        config.datatype = 'small';
        config.inPath = ['../data/ivy/all/'];
        config.isImageNet = false;
end


% 3rd party path: where the matconvnn locates
config.matconvv_path = '../../matconvnet-1.0-beta16-gpu/';

% parameter for synthesis
% nTileRow \times nTileCol defines the number of paralle chains
% right now, we currently support square chains, e.g. 2*2, 6*6, 10*10 ...

config.force_learn = true;
config.batch_size = 100;

switch exp_type
    case 'texture'
        config.nIteration = 5000;
        config.nTileRow = 1;
        config.nTileCol = 1;
        
        % parameters for net 1
        config.T = 30;
        config.Delta1 = 0.0023;
        config.Gamma1 = 0.02;
        config.refsig1 = config.Delta1 / 0.3;
        config.cap1 = 20/255;
        
        % parameters for net 2
        config.Delta2 = 0.1;
        config.Gamma2 = 0.000001;
        config.refsig2 = 1;
        config.s = 0.3;
        config.real_ref = 1;
        config.cap2 = 30;
    case 'object'
        config.nIteration = 3000;
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
        
        config.interp_type = 'both';
        config.n_pairs = 8;
        config.n_parsamp = 8;
end

run(fullfile(config.matconvv_path, 'matlab', 'vl_setupnn.m'));
addpath(genpath('./core/') )


% result file: no need to change
config.working_folder = ['./working/', config.categoryName, '_', config.model_type, '_', config.learn_scheme, '/'];
config.Synfolder = ['./synthesiedImage/', config.categoryName, '_', config.model_type, '_', config.learn_scheme, '/'];
config.figure_folder = ['./figure/', config.categoryName, '_', config.model_type, '_', config.learn_scheme, '/'];


% create directory
if ~exist('./working/', 'dir')
    mkdir('./working/')
end

if ~exist('./synthesiedImage/', 'dir')
   mkdir('./synthesiedImage/') 
end

if ~exist('./figure/', 'dir')
   mkdir('./figure/') 
end

if ~exist(config.Synfolder, 'dir')
   mkdir(config.Synfolder);
end

if ~exist(config.working_folder, 'dir')
    mkdir(config.working_folder);
end

if ~exist(config.figure_folder, 'dir')
    mkdir(config.figure_folder);
end

