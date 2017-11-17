function inter_out = interpolate_digits(net1,net2,ELM)
    config = gen_ADELM_config;
    %overwrite config
    config.refsig = 1;
    config.MH_eps = 0.05; %step size
    config.MH_type = 'RW'; 

    config.AD_temp = 1; % AD temperature parameter
    config.alpha = 12000; % AD magnetization strength
    config.max_AD_iter = 5000;  % max iters for AD trial
    config.AD_no_improve = 40; % consecutive iters to stop search
    config.dist_res = .35;

    config.bar_factor = 1.05; %grid search factor, greater than 1
    config.bar_AD_reps = 5;
 
    config.im_sz = [64,64,1];
    config.z_sz = [1,1,8];
    config.mean_im = net1.mean_im;
    
    inter_inds = [3,4,5,13,14]; % min indices for interpolation
    
    inter_out = [];
    pair_ij = [repmat(1:length(inter_inds),1,length(inter_inds))',...
        repelem(1:length(inter_inds),length(inter_inds))'];
    pair_ij = pair_ij(pair_ij(:,1)<pair_ij(:,2),:); 
    for ind = 1:size(pair_ij,1)
        ij = pair_ij(ind,:);
        config.alpha = 10000;
        [~,a_bar] = find_metastable_border(config,net1,net2,...
                        ELM.min_z(:,:,:,ij(1)),ELM.min_z(:,:,:,ij(2)));
        config.alpha = a_bar * config.bar_factor^2;
        reach = 0;
        while reach == 0
            AD_out = gen_AD(config,net1,net2,...
                        ELM.min_z(:,:,:,ij(1)),ELM.min_z(:,:,:,ij(2)));
            if AD_out.mem == 1
                inter_out{end+1} = AD_out;
                reach = 1;
            end
        end
    end
    save('inter_out_full.mat', 'inter_out');
end