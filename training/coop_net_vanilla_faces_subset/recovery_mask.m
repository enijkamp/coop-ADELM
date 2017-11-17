function loss = recovery_mask()

config = frame_config('face_30', 'mask', 'object');
addpath('./other/');
working_folder = config.working_folder;
Synfolder = config.Synfolder;
figure_folder = config.figure_folder;
inPath = config.inPath;
datatype = config.datatype;

model_file = './model/layer_01_iter_600_model.mat';
load(model_file);

config.working_folder = working_folder;
config.Synfolder = Synfolder;
config.figure_folder = figure_folder;
config.inPath = inPath;
config.datatype = datatype;
config.is_crop = false;
config_ori.is_preprocess = false;

config.T = 300;
config.nTileRow = 10;
config.nTileCol = 10;
config.Delta1 = 0.0003;
config.Delta2 = 0.03;


[imdb, getBatch] = create_imdb(config, net1);

config_ori = config;
config_ori.inPath = ['./data/', 'face_ori'];
config_ori.datatype = 'celebA';
config_ori.force_learn = true;
config_ori.is_crop = false;
config_ori.is_preprocess = false;

[imdb_ori, getBatch_ori] = create_imdb(config_ori, net1);

mean_img1 = net1.normalization.averageImage;
mean_img2 = net2.normalization.averageImage;

net1 = vl_simplenn_move(net1, 'gpu');
net2 = vl_simplenn_move(net2, 'gpu');

loss = zeros(1, config.T);


train = find(imdb.images.set==1);
num_batch = numel(1:config.batch_size:numel(train));
for t=1:config.batch_size:numel(train)
    fprintf('recovering: batch %3d/%3d: \n', ...
        fix(t/config.batch_size)+1, ceil(numel(train)/config.batch_size)) ;

    
    for s=1:1
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+config.batch_size-1, numel(train)) ;
        batch = train(batchStart : 1 * numlabs : batchEnd) ;
        
        [im, ~, mask] = getBatch(imdb, batch) ;
        im_ori = getBatch_ori(imdb_ori, batch);
        
        if t == 1
            I_ori = convert_syns_mat(config, mean_img1, im_ori);
            I_masked = convert_syns_mat(config, mean_img1, im);
            
            imwrite(I_ori,[config.Synfolder, 'original.png']);
            imwrite(I_masked,[config.Synfolder, 'masked.png']);
        end
        
        im = gpuArray(im);
        mask = gpuArray(mask);
        im_ori = gpuArray(im_ori);
        
        % recover from net2
        z = gpuArray(randn([config.z_sz, size(im, 4)], 'single'));
        res2 = [];
        for tt = 1:400
            res2 = vl_gan(net2, z, im, res2, ...
                'mask', mask, ...
                'accumulate', false, ...
                'disableDropout', true, ...
                'conserveMemory', true, ...
                'backPropDepth', +inf) ;
            
            z = z + config.Delta2 * config.Delta2 /2 /config.s /config.s* res2(1).dzdx ...
                - config.Delta2 * config.Delta2 /2 /config.refsig2 /config.refsig2* z;
            z = z + config.Delta2 * gpuArray(randn(size(z), 'single'));
            
            if t == 1 && (mod(tt, 10) == 1 || tt == 400)
                temp_mat = im;
                temp_mat(mask) = res2(end).x(mask);
                I_recovery = convert_syns_mat(config, mean_img2, gather(temp_mat));
                imwrite(I_recovery,[config.Synfolder, num2str(tt, 'recovered_net2_%04d'), '.png']);
                
                if tt == 400
                    for i = 1:size(temp_mat, 4)
                       img = gather(temp_mat(:,:,:,i));
                       img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
                       imwrite(img, [figure_folder, num2str(i, 'net2_%03d.png')]);
                       
                       img = gather(im_ori(:,:,:,i));
                       img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
                       imwrite(img, [figure_folder, num2str(i, 'original_%03d.png')]);
                    end
                end
            end
        end
        
        im(mask) = res2(end).x(mask);
        
        num_syn = size(im, 4);
        dydz = gpuArray(ones(config.dydz_sz1, 'single'));
        dydz = repmat(dydz, 1, 1, 1, num_syn);
        res = [];
        
        for tt = 1:config.T
            res = vl_simplenn(net1, im, dydz, res, ...
                'accumulate', false, ...
                'disableDropout', true, ...
                'conserveMemory', true, ...
                'backPropDepth', +inf) ;
            
            temp_mat = config.Delta1^2/2 * (res(1).dzdx - im / config.refsig1 /config.refsig1);% + ...
                %config.Delta1 * gpuArray(randn(size(im), 'single'));
            
            im(mask) = im(mask) + temp_mat(mask); 
            
            loss(tt) = loss(tt) + gather( mean(reshape(abs(im(mask) - im_ori(mask)), [], 1))) / num_batch / 2;
            
            if t == 1 && (mod(tt, 10) == 1 || tt == config.T)
                I_recovery = convert_syns_mat(config, mean_img1, gather(im));
                imwrite(I_recovery,[config.Synfolder, num2str(tt, 'recovered_net1_%04d'), '.png']);
                
                if tt == config.T
                    for i = 1:size(im, 4)
                       img = gather(im(:,:,:,i));
                       img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
                       imwrite(img, [figure_folder, num2str(i, 'net1_%03d.png')]);
                    end
                end
            end
        end        
        
        % mrf_l2
        save_dir = [config.figure_folder(1:end-1), '_mrfl2/'];
        error_mrfl2 = recover_mrf(im, im_ori, mask, save_dir, 1);
        
        % mrf_l1
        save_dir = [config.figure_folder(1:end-1), '_mrfl1/'];
        error_mrfl1 = recover_mrf(im, im_ori, mask, save_dir, 2);
        
        % nan1
        save_dir = [config.figure_folder(1:end-1), '_nan1/'];
        error_nan1 = recover_nan(im, im_ori, mask, save_dir, 1);
        
        % nan2
        save_dir = [config.figure_folder(1:end-1), '_nan2/'];
        error_nan2 = recover_nan(im, im_ori, mask, save_dir, 2);
        
        % nan3
        save_dir = [config.figure_folder(1:end-1), '_nan3/'];
        error_nan3 = recover_nan(im, im_ori, mask, save_dir, 3);
        
        % nan4
        save_dir = [config.figure_folder(1:end-1), '_nan4/'];
        error_nan4 = recover_nan(im, im_ori, mask, save_dir, 4);
        
        % nan5
        save_dir = [config.figure_folder(1:end-1), '_nan5/'];
        error_nan5 = recover_nan(im, im_ori, mask, save_dir, 5);
        
        % nan0
        save_dir = [config.figure_folder(1:end-1), '_nan0/'];
        error_nan0 = recover_nan(im, im_ori, mask, save_dir, 0);
        
        fid = fopen([config.Synfolder, 'error.txt'], 'w');
        fprintf(fid, '%.4f\n', min(loss));
        fprintf(fid, '%.4f\n', error_mrfl2);
        fprintf(fid, '%.4f\n', error_mrfl1);
        fprintf(fid, '%.4f\n', error_nan1);
        fprintf(fid, '%.4f\n', error_nan2);
        fprintf(fid, '%.4f\n', error_nan3);
        fprintf(fid, '%.4f\n', error_nan4);
        fclose(fid);
    end
end
net1 = vl_simplenn_move(net1, 'cpu');
net2 = vl_simplenn_move(net2, 'cpu');
% fprintf('The final error is %.2f\n', loss(end));
% figure;
% plot(1:config.T, loss, 'r-', 'LineWidth', 3);

