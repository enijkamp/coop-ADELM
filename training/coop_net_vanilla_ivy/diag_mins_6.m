function [] = diag_mins_6()

%% prep

% clear
clear all;
close all;
restoredefaultpath();

% prep
time = tic;
set(0,'DefaultTextInterpreter','none');
addpath(genpath('./core/'));
rng(123);

% matconvnet
Setup(1, 0);

% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
%assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
%assert(strcmp(dev.Name, 'Tesla P100-PCIE-16GB'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

% parpool
no_workers = 4;
if no_workers > 1
    delete(gcp('nocreate'));
    parpool('local', no_workers, 'IdleTimeout', Inf);
end

% set matconvnet for each worker
if no_workers > 1
    parfor i = 1:no_workers
        vl_setupnn();
    end
end

%% load
exp = 'ivy_dense_em_27_1';
load(['working/' exp '/layer_01_iter_200_model.mat']);

config = [];

%% config

net1 = vl_simplenn_move(net1, 'gpu');
net2 = vl_simplenn_move(net2, 'gpu');

% nets
config.des_net = net1;
config.gen_net = net2;

% output
config.im_folder = ['minima/' exp '/'];

% image and latent space
config.z_sz = [1,1,20];
config.im_sz = [64,64,3];
config.mean_im = single(zeros(config.im_sz));

% langevin
config.refsig1 = 1;

% local min search
config.MH_type = 'RW'; % random walk
config.MH_eps = 0.05; % step size
config.min_temp = .1; % temperature for min search
config.min_sweeps = 5000; % max number of sweeps during min search
config.min_no_improve = 30; % consecutive failed iters to stop search

% parameters for AD-ELM
config.nsteps = 500; % number of ELM iterations
config.num_mins = config.nsteps+1; % max number of basins on record
config.AD_heuristic = '1D_bar'; % 1D linear interpolation '1D_bar'

% attraction diffusion
config.max_AD_checks = 10; % number of minima for AD trials
config.AD_reps = 3; % number of AD attempts for each min 
config.max_AD_iter = 5000;  % max iters for AD trial

%% prep

% folders
if exist(config.im_folder, 'dir')
    rmdir(config.im_folder, 's');
end
mkdir(config.im_folder);

%% map with membership

% % start chain
% z = randn(config.z_sz,'single');
% [en,im] = get_gen_energy(config,config.des_net,config.gen_net,z);
% [min_z,min_im,min_ind,min_energy,ens,im_path,z_path,accept_rate] = find_gen_min(config,config.des_net,config.gen_net,z);
% 
% % make new record
% ELM = make_ELM_record(config,min_z);
% ELM = update_ELM_record(ELM,z,im,min_z,min_im,en,min_energy,1,1); 
% 
% % plot revision
% f1 = figure();
% imshow(uint8(im), []);
% f2 = figure();
% imshow(uint8(min_im), []);
% 
% % find mininma with membership
% viz_min_ims(ELM.min_ims,config.im_folder);
% for rep = 1:config.nsteps    
%     fprintf('\n');
%     fprintf('ELM Step %d of %d\n',rep,config.nsteps);
%     fprintf('----\n');
%     % find new min and classify in each ELM step
%     ELM = gen_ADELM_step(config,config.des_net,config.gen_net,ELM);
% 
%     %save results
%     viz_min_ims(ELM.min_ims,config.im_folder,1);
%     %save([config.ELM_folder,config.ELM_str,config.map_str],'ELM');
% end  

%% map without membership

% find minima
parfor rep = 1:config.nsteps    
    fprintf('Mininum %d of %d\n',rep,config.nsteps);
      
    % start chain
    z = randn(config.z_sz,'single');
    [en,im] = get_gen_energy(config,config.des_net,config.gen_net,z);
    [min_z,min_im,min_ind,min_energy,ens,im_path,z_path,accept_rate] = find_gen_min(config,config.des_net,config.gen_net,z);
    
    % save
    imwrite(min_im/256,[config.im_folder,'min_im',num2str(rep),'.png']);
end  

% time
iter_time = toc(time);
fprintf('\n');
fprintf('Total mapping time: %4.2f seconds \n',iter_time);


disp('done');

end