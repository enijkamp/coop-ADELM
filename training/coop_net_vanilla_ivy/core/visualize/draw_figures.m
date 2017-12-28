function [] = draw_figures(config, syn_mat, iter, mean_img, SSD, layer, prefix)

if nargin < 7
    prefix = '';
else
    prefix = [prefix, '_'];
end

[I_syn, syn_mat_norm] = convert_syns_mat(config, mean_img, syn_mat);

% for i = 1:size(syn_mat_norm, 4)
%     imwrite(syn_mat_norm(:,:,:,i), [config.figure_folder, prefix, num2str(layer, 'layer_%02d_'), num2str(i, '%03d.png')]);
% end

imwrite(I_syn, [config.Synfolder, prefix, num2str(layer, 'layer_%02d_'), num2str(iter, 'dense_original_%04d'), '.png']);

end