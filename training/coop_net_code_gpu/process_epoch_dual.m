function  [net1,net2,gen_mats,syn_mats,z,loss] = process_epoch_dual(opts, epoch, net1, net2, config)
% -------------------------------------------------------------------------

if config.use_gpu
    net1 = vl_simplenn_move(net1, 'gpu') ;
    net2 = vl_simplenn_move(net2, 'gpu') ;
end

fprintf('Training: epoch %02d', epoch) ;
fprintf('\n');
%randomize order of training images
imdb = config.imdb;
train_order = randperm(size(imdb,4));
batchNum = 0;
epoch_time = tic;

for t = 1:config.batchSize:size(imdb,4)
    batchNum = batchNum+1;
    fprintf('Epoch %d of %d, Batch %d of %d \n', epoch,config.nIteration,batchNum,ceil(size(imdb,4)/config.batchSize));
    batchTime = tic;

    batchStart = t;
    batchEnd = min(t+config.batchSize-1, size(imdb,4)) ;
    batch = train_order(batchStart : batchEnd) ;
    im = imdb(:,:,:,batch);   
    
    %% Step 1: Inference Network 2 -- generate Z
    z = randn([config.z_sz, config.num_syn], 'single');
    if config.use_gpu
        z = gpuArray(z);
        syn_mat = vl_gan(net2, z, [], [],...
                'accumulate', 0, ...
                'disableDropout', 0, ...
                'conserveMemory', opts.conserveMemory, ...
                'backPropDepth', opts.backPropDepth, ...
                'sync', opts.sync, ...
                'cudnn', opts.cudnn) ;
    else
        syn_mat = vl_gan_cpu(net2, z, [], [],...
                'accumulate', 0, ...
                'disableDropout', 0, ...
                'conserveMemory', opts.conserveMemory, ...
                'backPropDepth', opts.backPropDepth, ...
                'sync', opts.sync, ...
                'cudnn', opts.cudnn) ;
    end
    %disp(size(syn_mat(end).x));
    %syn_mats = floor(128*(syn_mat(end).x+1))-config.mean_im;
    syn_mats = floor(128*(syn_mat(end).x+1)) - repmat(config.mean_im, 1, 1, 1, config.num_syn);

    if config.use_gpu
        gen_mats = gather(syn_mats);
    else
        gen_mats = syn_mats;
    end
    
    % store synthesis from generator
    if config.use_gpu
        if t == 1
            if mod(epoch,10) == 0 || epoch == config.nIteration
                for i = 1:config.num_syn
                    imwrite((gen_mats(:,:,:,i)+config.mean_im)/256,[config.gen_im_folder,'gen_im',num2str(i),'.png']);
                end
                imwrite((gen_mats(:,:,:,1)+config.mean_im)/256,[config.gen_im_folder,'gen_im_epoch',num2str(epoch),'.png']);
            end
        end
    else
        for i = 1:config.num_syn
            imwrite((gen_mats(:,:,:,i)+config.mean_im)/256,[config.gen_im_folder,'gen_im',num2str(i),'.png']);
        end
    end
    
    %% Step 2: update generator mats by descriptor net
    % synthesize image according to current weights 
    if config.use_gpu
        syn_mats = langevin_dynamics_gpu(config, net1, syn_mats);
    else
        for syn_ind = 1:config.num_syn
        %new_im = single(find_gen_min(net1,net2,config,z(:,:,:,syn_ind)));
        %syn_mats(:,:,:,syn_ind) = new_im-config.mean_im;
        syn_mats(:,:,:,syn_ind) = langevin_dynamics(config,net1,syn_mats(:,:,:,syn_ind));
        %syn_mats(:,:,:,syn_ind) = find_local_min(net1,config,syn_mats(:,:,:,syn_ind));
        end
    end

    %% Step 3: learning net1
    if config.use_gpu
        dydz1 = gpuArray(zeros(config.dydz_sz1, 'single'));
        im = gpuArray(im);
    else
        dydz1 = zeros(config.dydz_sz1, 'single');
    end
    dydz1(net1.filterSelected) = net1.selectedLambdas;
    if config.use_gpu
        dydz1 = gpuArray(repmat(dydz1,1,1,1,size(im,4)));
    else
        dydz1 = repmat(dydz1,1,1,1,size(im,4));
    end
    
    res1 = [];
    res1 = vl_simplenn(net1, im, dydz1, res1, ...
        'accumulate', 0, ...
        'disableDropout', 0, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn);
    
    dydz_syn = zeros(config.dydz_sz1, 'single');
    dydz_syn(net1.filterSelected) = net1.selectedLambdas;
    if config.use_gpu
        dydz_syn = gpuArray(repmat(dydz_syn,1,1,1,config.num_syn));
    else
        dydz_syn = repmat(dydz_syn,1,1,1,config.num_syn);
    end
    
    res_syn = [];
    res_syn = vl_simplenn(net1, syn_mats, dydz_syn, res_syn, ...
        'accumulate', 0, ...
        'disableDropout', 0, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn);

    % gather and accumulate gradients across labs
    [net1, ~, loss] = accumulate_gradients1(opts, config.Gamma(epoch), size(im,4), net1, res1, res_syn, config);
    
    clear res;
    clear res_syn;
        
    if config.use_gpu
        syn_mats = gather(syn_mats);
        if t == 1
            if mod(epoch,10) == 0 || epoch == config.nIteration
                for k = 1:config.num_syn
                    imwrite((syn_mats(:,:,:,k)+config.mean_im)/256,[config.syn_im_folder,'syn_im',num2str(k),'.png']);
                end
                imwrite((syn_mats(:,:,:,1)+config.mean_im)/256,[config.syn_im_folder,'syn_im_epoch',num2str(epoch),'.png']);
            end
        end
    else
        for k = 1:config.num_syn
            imwrite((syn_mats(:,:,:,k)+config.mean_im)/256,[config.syn_im_folder,'syn_im',num2str(k),'.png']);
        end
    end

    % syn_mats = max(min(syn_mats+config.mean_im,255.99),0.01)/128 - 1;
    syn_mats = max(min(syn_mats+repmat(config.mean_im, 1, 1, 1, config.num_syn),255.99),0.01)/128 - 1;

    
    %% Step 4: Learning net2
    res2 = [];
    if config.use_gpu
        res2 = vl_gan(net2, z, syn_mats, res2, ...
            'accumulate', 0, ...
            'disableDropout', 0, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
    else
        res2 = vl_gan_cpu(net2, z, syn_mats, res2, ...
            'accumulate', 0, ...
            'disableDropout', 0, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
    end

    net2 = accumulate_gradients2(opts, config.Gamma2(epoch), net2, res2, config);     
    
    real_ref = std(z(:));
    config.real_ref = real_ref;  
    fprintf('max inferred z is %.2f, min inferred z is %.2f, and std is %.2f\n', max(z(:)), min(z(:)), config.real_ref)  
    
    batchTime = toc(batchTime);
    speed = 1/batchTime ;
    disp(batchTime);
    fprintf(' %Time: .2f s (%.1f data/s)', batchTime, speed) ;
    fprintf('\n') ;
end


if config.use_gpu
    net1 = vl_simplenn_move(net1, 'cpu') ;
    net2 = vl_simplenn_move(net2, 'cpu') ;
end

if config.use_gpu
    if mod(epoch,10) == 0 || epoch == config.nIteration
        save([config.trained_folder,'des_net.mat'],'net1');
        save([config.trained_folder,'gen_net.mat'],'net2');
    end
else
    save([config.trained_folder,'des_net.mat'],'net1');
    save([config.trained_folder,'gen_net.mat'],'net2');
end

% print learning statistics
epoch_time = toc(epoch_time) ;
speed = config.num_syn/epoch_time ;

fprintf(' %.2f s (%.1f data/s)\n', epoch_time, speed) ;

end

% -------------------------------------------------------------------------
function [net,res,loss] = accumulate_gradients1(opts, lr, batchSize, net, res, res_syn, config)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets1;

loss = [];

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;

        if isfield(net.layers{l}, 'weights')
            
            gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                        (1 / config.num_syn) * res_syn(l).dzdw{j}) / net.numFilters(l);
                    
            if max(abs(gradient_dzdw(:))) > 20 %10
                gradient_dzdw = gradient_dzdw / max(abs(gradient_dzdw(:))) * 20;
            end
            
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR*net.layers{l}.momentum{j};
            
            loss = [loss, gather(mean(abs(gradient_dzdw(:))))];
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('\n layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
loss = mean(loss);
end
 

% -------------------------------------------------------------------------
function [net, res] = accumulate_gradients2(opts, lr, net, res, config)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets2;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        if isfield(net.layers{l}, 'weights')
            % gradient descent
            gradient_dzdw = (1 / config.s / config.s)* res(l).dzdw{j};
            
            max_val = max(abs(gradient_dzdw(:)));
            
            if max_val > config.cap2
                gradient_dzdw = gradient_dzdw / max_val * config.cap2;
            end
  
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            %             net.layers{l}.momentum{j} = gradient_dzdw;
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR *net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('Net2: layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
end

