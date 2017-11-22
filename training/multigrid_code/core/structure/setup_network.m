function [net] = setup_network(net)

net.normalization.imageSize = [net.h_res, net.h_res, 3];
net.normalization.averageImage = zeros(net.normalization.imageSize, 'single');

img = randn(net.normalization.imageSize, 'single');
net = vl_simplenn_move(net, 'gpu') ;
res = vl_simplenn(net, gpuArray(img));
net = vl_simplenn_move(net, 'cpu');
net.dydz_sz = size(res(end).x);

net.numFilters = zeros(1, length(net.layers));
for l = 1:length(net.layers)
    if isfield(net.layers{l}, 'weights')
        sz = size(res(l+1).x);
        net.numFilters(l) = sz(1) * sz(2);
    end
end

net.layer_sets = numel(net.layers):-1:1;

clear res;
clear img;
end