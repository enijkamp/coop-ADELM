function [] = ivy3_elm_27_1(no_workers, pool_name_comet)

%% exp
exp_id = '27_1';

%% setup
addpath(genpath('../../main/'));
exp_time = tic;
rng(123);

%% load config
ELM_str = ['ivy3/' exp_id '/'];
net_file_str = 'layer_01_iter_200_model.mat';
config = gen_ADELM_config(ELM_str,ELM_str,net_file_str); % adjust image and z size

%% pool init
if nargin > 0
    config.no_workers = no_workers;
end

% pool with no timeout to keep paths
if(config.no_workers > 2 && nargin > 0)
    pool = parpool(pool_name_comet, config.no_workers, 'IdleTimeout', Inf);
elseif(config.no_workers > 1)
    delete(gcp('nocreate'));
    pool = parpool('local', config.no_workers, 'IdleTimeout', Inf);
end

% set paths
addpath(genpath('../../matconvnet-1.0-beta16/'));
vl_setupnn();
if(config.no_workers > 2)
    vl_compilenn();
end

% set matconvnet for each worker
if(config.no_workers > 1)
    parfor i = 1:config.no_workers
        vl_setupnn();
    end
end


%% config override

%ELM params
config.nsteps = 1000;
config.num_mins = config.nsteps+1;
%AD params
config.alpha = 1000000;
config.max_AD_checks = 10;
config.AD_reps = 3;
config.AD_quota = 1;
%AD extrema params
config.extrema_factor = 1.08; %grid search factor (greater than 1)
config.extrema_steps = 50;
config.max_extrema_checks = 5;
%consolidate params
config.max_consolidate_checks = 5;
config.consolidate_reps = 5;
config.consolidate_quota = 1;

%number of different magnetization strengths to be tested
num_exps = 10;
%upper and lower resolution for scale space boundary
alpha_init = [config.alpha,config.alpha];

%run experiment
run_ELM_experiment(config,num_exps,alpha_init);

%% pool shutdown
if(config.no_workers > 1)
    delete(pool);
end

%% done
exp_time = toc(exp_time);
fprintf('Total Experiment Time: %4d hours %4.2f minutes.\n',floor(exp_time/3600), mod(exp_time/60,60));

end