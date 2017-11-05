function [] = ivy_all_4_1_2_cont(no_workers, pool_name_comet)

%% exp
% (4_1_2): continue 5000 steps
exp_id = '4_1_2_cont';

%% setup
addpath(genpath('../../main/'));
exp_time = tic;
rng(123);


%% load config
out_file_str = ['ivy/all/' exp_id '/'];
ELM_test = load('../../maps/ivy/all/4_1/ELM_4_1_2_exp.mat');
ELM_test.ELM_str = out_file_str;
ELM_test.config.map_str = ['ELM_' exp_id '_exp.mat'];
ELM_test.config.nsteps = 5000;

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