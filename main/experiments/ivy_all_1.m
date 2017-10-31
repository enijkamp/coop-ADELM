function [] = ivy_all_1()

%% setup
exp_time = tic;
rng(123);

addpath(genpath('../../matconvnet-1.0-beta16/'));
addpath(genpath('../../main/'));
vl_setupnn();
vl_compilenn();


%% config
file_str = 'ivy/512/';
config = gen_ADELM_config(file_str);


%% run
no_workers = 24;

% pool with no timeout to keep paths
%pool = gcp('nocreate');
%delete(pool);
pool = parpool('pool1', no_workers, 'IdleTimeout', Inf);

% set path for each worker
parfor i = 1:no_workers
    vl_setupnn();
end

run_ELM_experiment(config,num_exps,alpha_init);


%% shutdown
delete(pool);

exp_time = toc(exp_time);
fprintf('Total Experiment Time: %4d hours %4.2f minutes.\n',floor(exp_time/3600), mod(exp_time/60,60));

end