function [] = ivy_all_4_1(no_workers, pool_name_comet)

%% setup
addpath(genpath('../../main/'));
exp_time = tic;
rng(123);


%% load config
%set up config
file_str = 'ivy/all/4_1/';
net_str = 'nets.mat';
config = gen_ADELM_config(file_str,net_str);


%% pool init
if nargin > 0
    config.no_workers = no_workers;
end

% pool with no timeout to keep paths
if(config.no_workers > 2)
    pool = parpool(pool_name_comet, config.no_workers, 'IdleTimeout', Inf);
elseif(config.no_workers > 1)
    delete(gcp('nocreate'));
    pool = parpool('local', config.no_workers, 'IdleTimeout', Inf);
end

% set paths
addpath(genpath('../../matconvnet-1.0-beta16/'));
vl_setupnn();
if(config.no_workers > 2)
    %vl_compilenn();
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

alpha_init = [1000000,1000000];
nsteps = config.nsteps;
max_AD_checks = config.max_AD_checks;
AD_reps = config.AD_reps;
AD_quota = config.AD_quota;
%load config for ELM experiments. only alpha and map_str will change
%find lower (infinite mins) and upper (single basin) bounds 
% of magnetization strength for ADELM
disp('# (0) find_AD_extrema');
tic_ext = tic;
[min_out,max_out] = find_AD_extrema(config,alpha_init);    
fprintf('# (0) find_AD_extrema: %4d hours %4.2f minutes.\n',floor(toc(tic_ext)/3600), mod(toc(tic_ext)/60,60));
    
%%line below updated
config.num_exps = 10;
alphas = exp(linspace(log(min_out.alpha),log(max_out.alpha),config.num_exps));
%%%%%%

disp('# (1) burnin');
tic_burn = tic;
config.alpha = alphas(6);
config.map_str = 'ELM_4_1_burnin.mat';
config.nsteps = floor(nsteps/2);
config.max_AD_checks = floor(max_AD_checks/2);
config.AD_quota = 1;
config.AD_reps = 1;
ELM_burnin = gen_ADELM([],config);
fprintf('# (1) burnin: %4d hours %4.2f minutes.\n',floor(toc(tic_burn)/3600), mod(toc(tic_burn)/60,60));

disp('# (2) consolidate_minima');
tic_cons = tic;
ELM_test = consolidate_minima(ELM_burnin);
ELM_test.config.min_out = min_out;
ELM_test.config.max_out = max_out;
ELM_test.config.map_str = 'ELM_4_1_exp.mat';
ELM_test.config.nsteps = nsteps;
ELM_test.config.max_AD_checks = max_AD_checks;
ELM_test.config.AD_quota = AD_quota;
ELM_test.config.AD_reps = AD_reps;
fprintf('# (2) consolidate_minima: %4d hours %4.2f minutes.\n',floor(toc(tic_cons)/3600), mod(toc(tic_cons)/60,60));


%% run
disp('# (3) gen_ADELM');
tic_gen = tic;
gen_ADELM(ELM_test);
fprintf('# (3) gen_ADELM: %4d hours %4.2f minutes.\n',floor(toc(tic_gen)/3600), mod(toc(tic_gen)/60,60));


%% pool shutdown
if(config.no_workers > 1)
    delete(pool);
end

%% done
exp_time = toc(exp_time);
fprintf('Total Experiment Time: %4d hours %4.2f minutes.\n',floor(exp_time/3600), mod(exp_time/60,60));

end