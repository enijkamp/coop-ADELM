function [] = ivy_all_4_2_1_barmat_1(no_workers, pool_name)

tot_time = tic;

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
in_file_str = 'ivy/all/4_1/';
out_file_str = ['ivy/all/4_1_2/'];
config = gen_ADELM_config(in_file_str,out_file_str);

%% run
for i = 1:10
    disp(['exp ' num2str(i)]);
    exp_time = tic;

    %load complete ELM experiment
    load([config.ELM_folder,out_file_str,'ELM_exp',...
            num2str(i),'.mat']);
        
    %adjust config
    ELM.config.bar_alpha = ELM.config.alpha * 2;
    ELM.config.max_bar_checks = 15;
    ELM.config.bar_AD_reps = 5;
    ELM.config.bar_temp = 20;
    
    %get barrier mat
    bar_mat = get_barrier_mat_quick(ELM);
    
    %save results
    if ~exist([ELM.config.tree_folder,out_file_str],'dir') mkdir([ELM.config.tree_folder,out_file_str]); end
    save([ELM.config.tree_folder,out_file_str,'/bar_mat',num2str(i),'.mat'],'bar_mat');
    
    % time
    exp_time = toc(exp_time);
    fprintf('Experiment Time: %4d hours %4.2f minutes \n',...
                         floor(exp_time/3600), mod(exp_time/60,60));
end

%% shutdown
delete(pool);

tot_time = toc(tot_time);
fprintf('Total Experiment Time: %4d hours %4.2f minutes \n',...
                     floor(tot_time/3600), mod(tot_time/60,60));
        

end
