function [ config ] = load_images( config )

    %load data
    if strcmp(config.file_str,'digits/')
        config.digits = 0:9; %digits to be used in the model
        config.set = 'test'; %train, test, both
        %read MNIST and save results to config
        [imdb,im_mat, im_labs, mean_im] = read_MNIST(config);
        config.imdb = single(imdb);
        config.imdb_mean = imresize(mean_im,[config.im_size,config.im_size]);
        config.im_mat = im_mat;
        config.im_labs = im_labs;
        %config.mean_im = single(zeros(config.im_size));
        config.mean_im = config.imdb_mean;
        config.imdb = config.imdb - repmat(config.mean_im,[1,1,1,size(config.imdb,4)]);
    else
        %get texture patches from original image if config.process_ims=1
        if config.process_ims == 1
            process_ims([config.inPath,config.process_str],...
                [config.inPath,config.file_str],config.resize_factor,...
                    config.im_size,config.num_patch);
        end
        
        %load images from folder
        files = dir([config.inPath,config.file_str,'*.png']);
        imdb = zeros(config.im_size,config.im_size,3,length(files));
        for i = 1:length(files)
            imdb(:,:,:,i) = imread([config.inPath,config.file_str,files(i).name]);
        end

        if strcmp(config.file_str(1:3),'ivy')
            config.mean_im = single(sum(imdb,4)/size(imdb,4));
        elseif strcmp(config.file_str,'escher/')
            config.mean_im = single(128*ones(config.im_size,config.im_size,3));
        end
        if config.substract_mean
            config.imdb = single(imdb - repmat(config.mean_im,1,1,1,size(imdb,4))); 
        else
            config.mean_im = zeros(config.im_size,config.im_size,3);
            config.imdb = single(imdb);
        end
    end


end

