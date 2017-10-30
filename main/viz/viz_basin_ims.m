function [ord,inds,ens]= viz_basin_ims(ELM,index,im_folder,config,max,keep)
    if isempty(im_folder), im_folder = config.im_folder; end
    if nargin<6 || isempty(keep), delete([im_folder,'*.png']); end
    if nargin<5 || isempty(max), max = sum(ELM.min_ID_path==index); end
    inds = find(ismember(ELM.min_ID_path,index));
    pperm = randperm(length(inds));
    inds = inds(pperm(1:max));
    ens = ELM.min_en_path(inds);
    [~,ord]=sort(ens);
    %disp(length(ord));
    imwrite(ELM.min_ims(:,:,:,index)/256,[im_folder,...
            'basin',num2str(index),'_min.png']);
        
    for i = 1:length(ord)
        imwrite(ELM.min_im_path(:,:,:,inds(ord(i)))/256,[im_folder,...
            'basin',num2str(ELM.min_ID_path(inds(ord(i)))),'_',num2str(i),'.png']);
    end
end