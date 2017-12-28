function [net1, net2, config] = train_model_dual(config, net1, net2, imdb, getBatch, layer)

opts.batchSize = config.batch_size ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = config.gpus; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false;
opts.numFetchThreads = 8;
opts.cudnn = true ;
opts.weightDecay = 0.0001 ; %0.0001
opts.momentum = 0.5;
opts.memoryMapFile = fullfile(config.working_folder, 'matconvnet.bin') ;

if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

opts.batchSize = min(opts.batchSize, numel(opts.train));
opts.numEpochs = config.nIteration;

net1 = initialize_momentum(net1);
net2 = initialize_momentum(net2);

interval = 10; %ceil(opts.numEpochs / 50);
loss1 = zeros(opts.numEpochs, 1);
loss2 = zeros(opts.numEpochs, 1);

mean_img1 = net1.normalization.averageImage;
mean_img2 = net2.normalization.averageImage;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate'))
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% -------------------------------------------------------------------------
%                           Train and validate
% -------------------------------------------------------------------------

model_file = [config.working_folder, num2str(layer, 'layer_%02d'), '_iter_',...
    num2str(opts.numEpochs) ,'_model.mat'];

if exist(model_file, 'file') && config.force_learn == false
    load(model_file);
else
    h = figure('pos', [100, 100, 1000, 400]);
%% added
    rng(123);
    for epoch=1:opts.numEpochs
        
        %% train
        fprintf('Iteration %d / %d\n', epoch, opts.numEpochs);
        train = opts.train;
        [net1, net2, config, syn_mats, gen_mats, z_mats, loss2(epoch)] = process_epoch_dual(opts, config, getBatch, epoch, train, imdb, net1, net2);
        
        %% loss
        % loss 1: l2 distance between descriptor revision and generator synthesis after learning
        % loss 2: mean of gradients of descriptor net (if training samples equal synthesis samples, then gradient should be 0)
        loss1(epoch) = compute_loss(config, opts, syn_mats, net2, z_mats);
        save([config.working_folder,'loss.mat'],'loss1','loss2');
        fprintf('loss1 = %.4f.\nloss2 = %.4f.\n', loss1(epoch), loss2(epoch));
        clf(h);
        hold on;
        subplot(1,2,1); plot(1:epoch, loss1(1:epoch), 'r');
        subplot(1,2,2); plot(1:epoch, loss2(1:epoch), 'b');
        hold off;
        drawnow;
        saveas(h, [config.working_folder,'loss.png']);
        
        %% minima
        if config.find_minima && epoch >= config.find_minima_epoch && epoch ~= 1 && (mod(epoch - 1, 10) == 0 || epoch == opts.numEpochs)
            
            % metropolis-hastings to find minima
            [ min_ims ] = find_gen_mins(config.z_sz, size(config.mean_im), 8^2, net1, net2);
            [I_min_ims, ~] = convert_syns_mat(config, zeros(size(config.mean_im), 'single'), min_ims);
            out_file = [config.Synfolder, prefix, num2str(layer, 'layer_%02d_'), num2str(iter, 'dense_original_%04d_minima'), '.png'];
            imwrite(I_min_ims, out_file);
        end

        %% synthesis
        if mod(epoch - 1, interval) == 0 || epoch == opts.numEpochs
            
            % (1) ancestral samples from generator net2
            gen_mat = gen_mats{1};
            [I_syn, ~] = convert_syns_mat(config, config.mean_im, gen_mat);
            out_file = [config.Synfolder, num2str(epoch, '%04d'), '_1_net2', '.png'];
            imwrite(I_syn, out_file);

            % (2) revisions from descriptor net1
            syn_mat = syn_mats{1};
            [I_syn, ~] = convert_syns_mat(config, config.mean_im, syn_mat);
            out_file = [config.Synfolder, num2str(epoch, '%04d'), '_2_net1', '.png'];
            imwrite(I_syn, out_file);
            
            % (3) revised samples from generator net2
            z = z_mats{1};   
            gen_mats_2 = generate_imgs(opts, net2, z);
            if ~config.normalize_images
                gen_mats_2 = floor(128*(gen_mats_2+1));
            end
            [I_syn, ~] = convert_syns_mat(config, zeros(size(config.mean_im), 'single'), gen_mats_2);
            out_file = [config.Synfolder, num2str(epoch, '%04d'), '_3_net2', '.png'];
            imwrite(I_syn, out_file);
        end
        
        %% model
        if mod(epoch - 1, 50) == 0 || epoch == opts.numEpochs
            cell_idx = randperm(numel(z_mats), 1);
            z = z_mats{cell_idx};
            interpolator(config, net2, z, epoch);

            model_file = [config.working_folder, num2str(layer, 'layer_%02d'), '_iter_',num2str(epoch) ,'_model.mat'];
            %save(model_file, 'net1', 'net2', 'z_mats', 'syn_mats', 'config');
            save(model_file, 'net1', 'net2', 'config');

            saveas(h, [config.working_folder, num2str(layer, 'layer_%02d_'), '_iter_',num2str(epoch) ,'_error.fig']);
            saveas(h, [config.working_folder, num2str(layer, 'layer_%02d_'), '_iter_',num2str(epoch) ,'_error.png'])
        end
    end
end
end

function loss = compute_loss(config, opts, syn_mats, net_cpu, z_mats)
net = vl_simplenn_move(net_cpu, 'gpu') ;
loss = 0;
res = [];
for i=1:numel(syn_mats)
    syn_mat = syn_mats{i};
    z = z_mats{i};

    res = vl_gan(net, gpuArray(z), gpuArray(syn_mat), res, ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    
%% added
    gen_mat = res(end).x;
    if ~config.normalize_images
        gen_mat = floor(128*(gen_mat+1)) - repmat(config.mean_im, 1, 1, 1, config.num_syn);
    end
    
    loss = loss + gather( mean(reshape(sqrt((gen_mat - syn_mat).^2), [], 1)));
end
end

function imgs = generate_imgs(opts, net_cpu, z)
net = vl_simplenn_move(net_cpu, 'gpu') ;
res = vl_gan(net, gpuArray(z), [], [], ...
    'accumulate', false, ...
    'disableDropout', true, ...
    'conserveMemory', opts.conserveMemory, ...
    'backPropDepth', opts.backPropDepth, ...
    'sync', opts.sync, ...
    'cudnn', opts.cudnn) ;
imgs = gather(res(end).x);
end

function net = initialize_momentum(net)
for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights) ;
        for j=1:J
            net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
        end
        if ~isfield(net.layers{i}, 'learningRate')
            net.layers{i}.learningRate = ones(1, J, 'single') ;
        end
        if ~isfield(net.layers{i}, 'weightDecay')
            net.layers{i}.weightDecay = ones(1, J, 'single') ;
        end
    end
end
end