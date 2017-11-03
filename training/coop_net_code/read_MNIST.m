function [imdb, im_mat, im_labs, mean_im, fn] = read_MNIST(config)
    %if ~ismember(config.set,['train','test','both']), config.set = 'test'; end
    fn = @(imdb,batch)getBatch(imdb,batch);
    if strcmp(config.set,'train')
        fID = fopen([config.inPath 'train_ims']);
        im_bin = fread(fID);
        fclose(fID);
        im_mat = reshape(im_bin(17:end),28*28,60000)';
         
        fID = fopen([config.inPath 'train_labs']);
        lab_bin = fread(fID);
        fclose(fID);
        im_labs = lab_bin(9:end);
        
    end
    
    if strcmp(config.set,'test')
        fID = fopen([config.inPath 'test_ims']);
        im_bin = fread(fID);
        fclose(fID);
        im_mat = reshape(im_bin(17:end),28*28,10000)';
        im_mat = im_mat(1:5000,:); 
        fID = fopen([config.inPath 'test_labs']);
        lab_bin = fread(fID);
        fclose(fID);
        im_labs = lab_bin(9:end);
        im_labs = im_labs(1:5000);
    end
        
    if strcmp(config.set,'both')
        fID = fopen([config.inPath 'train_ims']);
        im_bin = fread(fID);
        fclose(fID);
        im_mat = reshape(im_bin(17:end),28*28,60000)';
         
        fID = fopen([config.inPath 'train_labs']);
        lab_bin = fread(fID);
        fclose(fID);
        im_labs = lab_bin(9:end);
        
        fID = fopen([config.inPath 'test_ims']);
        im_bin = fread(fID);
        fclose(fID);
        im_mat = [im_mat; reshape(im_bin(17:end),28*28,10000)'];
         
        fID = fopen([config.inPath 'test_labs']);
        lab_bin = fread(fID);
        fclose(fID);
        im_labs = [im_labs; lab_bin(9:end)];
        
    end
    
    imdb = zeros(config.im_size,config.im_size,1,sum(ismember(im_labs,config.digits)),'single');
    mean_im = reshape(mean(im_mat(ismember(im_labs,config.digits),:)),28,28)';
    mean_im = imresize(mean_im,[config.im_size,config.im_size]);
    num_ims = 0;
    for i = 1:length(im_labs)
        if ismember(im_labs(i),config.digits)
            num_ims = num_ims+1;
            imdb(:,:,1,num_ims) = imresize(reshape(im_mat(i,:),28,28)',[config.im_size,config.im_size]);
            %imdb(:,:,1,num_ims) = imresize(reshape(im_mat(i,:),28,28)'-mean_im,[config.im_size,config.im_size]);
        end
    end
    
    im_mat = im_mat(ismember(im_labs,config.digits),:);
    im_labs = im_labs(ismember(im_labs,config.digits))';
end

function im = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb(:,:,:,batch) ;
end