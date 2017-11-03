function process_ims(process_str,save_str,factor,patch_size,num_patch)
    delete([save_str,'*.png']);
    orig_im = imread(process_str);
    for i = 1:length(factor)
        im = imresize(orig_im,factor(i));
        for j = 1:num_patch
            start_pix = randi(min(size(im,1),size(im,2))-patch_size+1);
            imwrite(im(start_pix:(start_pix+patch_size-1),...
                start_pix:(start_pix+patch_size-1),:),[save_str,num2str((i-1)*num_patch+j),'.png']);
        end
    end
end