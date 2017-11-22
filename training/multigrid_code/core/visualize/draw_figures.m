function [] = draw_figures(opts, syn_mat, iter, mean_img, SSD, prefix)

if nargin < 6
    prefix = '';
else
    prefix = [prefix, '_'];
end

opts.sx = size(syn_mat, 1);
opts.sy = size(syn_mat, 2);

[I_syn, syn_mat_norm] = convert_syns_mat(opts, mean_img, syn_mat);

% for i = 1:size(syn_mat_norm, 4)
%     imwrite(syn_mat_norm(:,:,:,i), [opts.figure_folder, prefix, num2str(i, '%03d.png')]);
% end

imwrite(I_syn, [opts.Synfolder, prefix, num2str(iter, '%04d'), '.png']);

% generate gif
% im = im2uint8(I_syn);
% [imind,cm] = rgb2ind(im,256);
% if iter == 1
%     imwrite(imind, cm, [opts.Synfolder, 'animation', '.gif'], 'DelayTime', 0.10, 'Loopcount', inf);
% elseif iter > 1
%     imwrite(imind, cm, [opts.Synfolder, 'animation', '.gif'], 'WriteMode', 'append', 'DelayTime', 0.10);
% end

if ~isempty(SSD)
    h = figure;
    plot(1:iter, SSD(1:iter), 'r', 'LineWidth', 3);
    axis([min(iter, 1), iter+1, 0,  max(SSD(min(iter, 1):end)) * 1.2]);
    title('Reconstruction error')
    saveas(h, [opts.working_folder, 'iter',...
                    num2str(iter), '_', prefix, '_error.png'])
end

end