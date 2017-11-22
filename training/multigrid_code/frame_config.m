function config = frame_config(category)

% we only support gpu
config.gpus = 1;

if nargin < 1
    category = 'cat';
end

if nargin < 2
    h_res = 64;
end
% category name
config.categoryName = category;

% image path: where the dataset locates
config.datatype = 'small';
config.inPath = ['../image/',  config.categoryName '/'];
config.isImageNet = false;

% 3rd party path: where the matconvnn locates
config.matconvv_path = '../../matconvnet-1.0-beta16-gpu/';

% parameter for synthesis
% nTileRow \times nTileCol defines the number of paralle chains
% right now, we currently support square chains, e.g. 2*2, 6*6, 10*10 ...
config.force_learn = true;
config.batch_size = 100;

config.numEpochs = 1250; % 1250
config.nTileRow = 10;
config.nTileCol = config.nTileRow;

% parameters for net 1
config.T = 30; % 30
config.Delta = 0.3;%0.2
config.Gamma = 0.3; % 0.2
config.refsig = 50;%10
config.cap = 1; % 5

config.sx = 64;
config.sy = 64;
config.res = [1, 4, 16, 64];

run(fullfile(config.matconvv_path, 'matlab', 'vl_setupnn.m'));
addpath(genpath('./core/') )


% result file: no need to change
config.working_folder = ['./working/', config.categoryName,'/',];
config.Synfolder = ['./synthesiedImage/', config.categoryName, '/'];
config.figure_folder = ['./figure/', config.categoryName, '/'];

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

