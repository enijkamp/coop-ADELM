function [] = ivy_all_4_2_1_barmat_1(no_workers, pool_name)

%% exp
exp_id = '4_1_2';
in_file_str = 'ivy/all/4_1/';

%% matconvnet
addpath(genpath('../../matconvnet-1.0-beta16/'));
addpath(genpath('../../main/'));
vl_setupnn();
vl_compilenn();

%% parpool
pool = parpool(pool_name, no_workers, 'IdleTimeout', Inf);
parfor i = 1:no_workers
    vl_setupnn();
end

%% config
out_file_str = ['ivy/all/' exp_id '/'];
config = gen_ADELM_config(in_file_str,out_file_str);

%% run
%load complete ELM experiment
load([config.ELM_folder,in_file_str,'ELM_',exp_id,'_exp.mat']);

%adjust config
ELM.config.bar_alpha = ELM.config.alpha * 2;
ELM.config.max_bar_checks = 15;
ELM.config.bar_AD_reps = 5;
ELM.config.bar_temp = 20;

%get barrier mat
tot_time = tic;
bar_mat = get_barrier_mat_quick(ELM);

%save results
if ~exist([ELM.config.tree_folder,in_file_str],'dir') mkdir([ELM.config.tree_folder,in_file_str]); end
save([ELM.config.tree_folder,in_file_str,'/bar_mat_',exp_id,'.mat'],'bar_mat');
    
%% shutdown
delete(pool);

tot_time = toc(tot_time);
fprintf('Total Experiment Time: %4d hours %4.2f minutes \n',...
                     floor(tot_time/3600), mod(tot_time/60,60));
        

end
