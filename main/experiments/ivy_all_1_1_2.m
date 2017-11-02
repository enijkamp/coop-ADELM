function [] = ivy_all_1_1_2(no_workers, pool_name_comet)

%% exp
% (1_1_2): refine (1_1) with new alpha values
exp_id = '1_1_2';

%% setup
addpath(genpath('../../main/'));
exp_time = tic;
rng(123);


%% load config
in_file_str = 'ivy/all/1_1/';
out_file_str = ['ivy/all/' exp_id '/'];
config = gen_ADELM_config(in_file_str,out_file_str);


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

nsteps = config.nsteps;
max_AD_checks = config.max_AD_checks;
AD_reps = config.AD_reps;
AD_quota = config.AD_quota;
%load config for ELM experiments. only alpha and map_str will change
%find lower (infinite mins) and upper (single basin) bounds 
% of magnetization strength for ADELM
%%line below updated
alpha=280000;
%%%%%%

disp('# (1) burnin');
tic_burn = tic;
config.alpha = alpha;
config.map_str = ['ELM_' exp_id '_burnin.mat'];
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
ELM_test.config.map_str = ['ELM_' exp_id '_exp.mat'];
ELM_test.config.nsteps = nsteps;
ELM_test.config.max_AD_checks = max_AD_checks;
ELM_test.config.AD_quota = AD_quota;
ELM_test.config.AD_reps = AD_reps;
fprintf('# (2) consolidate_minima: %4d hours %4.2f minutes.\n',floor(toc(tic_cons)/3600), mod(toc(tic_cons)/60,60));

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