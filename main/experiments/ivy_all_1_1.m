function [] = ivy_all_1_1()

%% setup
exp_time = tic;
rng(123);

addpath(genpath('../../matconvnet-1.0-beta16/'));
addpath(genpath('../../main/'));
vl_setupnn();
vl_compilenn();


%% config
%set up config
file_str = 'ivy/all/';
net_str = 'nets1_1.mat';
config = gen_ADELM_config(file_str,net_str);

%ELM params
config.nsteps = 1000;
config.num_mins = config.nsteps+1;
%AD params
config.max_AD_checks = 10;
config.AD_reps = 3;
config.AD_quota = 1;
%AD extrema params
config.extrema_factor = 1.08; %grid search factor (greater than 1)
config.extrema_steps = 40;
config.max_extrema_checks = 5;
%consolidate params
config.max_consolidate_checks = 5;
config.consolidate_reps = 5;
config.consolidate_quota = 1;

alpha_init = 1000000;
nsteps = config.nsteps;
max_AD_checks = config.max_AD_checks;
AD_reps = config.AD_reps;
AD_quota = config.AD_quota;
%load config for ELM experiments. only alpha and map_str will change
%find lower (infinite mins) and upper (single basin) bounds 
% of magnetization strength for ADELM
[min_out,max_out] = find_AD_extrema(config,alpha_init);    

%%line below updated
alphas = exp(linspace(log(min_out.alpha),log(max_out.alpha),num_exps));
%%%%%%

config.alpha = alphas(6);
config.map_str = 'ELM_1_1_burnin.mat';
config.nsteps = floor(nsteps/2);
config.max_AD_checks = floor(max_AD_checks/2);
config.AD_quota = 1;
config.AD_reps = 1;
ELM_burnin = gen_ADELM([],config);
ELM_test = consolidate_minima(ELM_burnin);
ELM_test.config.map_str = 'ELM_1_1_exp.mat';
ELM_test.config.nsteps = nsteps;
ELM_test.config.max_AD_checks = max_AD_checks;
ELM_test.config.AD_quota = AD_quota;
ELM_test.config.AD_reps = AD_reps;


%% run
% pool with no timeout to keep paths
%pool = gcp('nocreate');
%delete(pool);
if(config.no_workers > 1)
    pool = parpool('pool1', config.no_workers, 'IdleTimeout', Inf);
end

% set path for each worker
parfor i = 1:config.no_workers
    vl_setupnn();
end

gen_ADELM(ELM_test);


%% shutdown
if(config.no_workers > 1)
    delete(pool);
end

exp_time = toc(exp_time);
fprintf('Total Experiment Time: %4d hours %4.2f minutes.\n',floor(exp_time/3600), mod(exp_time/60,60));

end