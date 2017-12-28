function [] = print_nets( net1, net2, im_size, z_size )

disp('# net1 #');
dydz_sz = [im_size,im_size,1];
for l = 1:numel(net1.layers)
    if strcmp(net1.layers{l}.type, 'conv')
        disp(['layer ' net1.layers{l}.name]);
        weights_sz = size(net1.layers{l}.weights{1});
        fprintf('  in  -> %4.0f %4.0f %4.0f\n', dydz_sz(1), dydz_sz(2), weights_sz(3));
        f_sz = size(net1.layers{l}.weights{1});
        pads = [net1.layers{l}.pad(1)+net1.layers{l}.pad(2), net1.layers{l}.pad(3)+net1.layers{l}.pad(4)];
        dydz_sz(1:2) = floor((dydz_sz(1:2) + pads - f_sz(1:2)) ./ net1.layers{l}.stride)+1;
        if size(weights_sz, 2) >= 4
            weights_out = weights_sz(4);
        else
            weights_out = 1;
        end
        fprintf('  out -> %4.0f %4.0f %4.0f\n', dydz_sz(1), dydz_sz(2), weights_out);
    end
end
disp(' ');

disp('# net2 #');
dydz_sz = [z_size(1:2), 1];
for l = 1:numel(net2.layers)
    if strcmp(net2.layers{l}.type, 'convt')
        disp(['layer ' net2.layers{l}.name]);
        weights_sz = size(net2.layers{l}.weights{1});
        fprintf('  in  -> %4.0f %4.0f %4.0f\n', dydz_sz(1), dydz_sz(2), weights_sz(4));
        f_sz = size(net2.layers{l}.weights{1});
        crops = [net2.layers{l}.crop(1)+net2.layers{l}.crop(2), net2.layers{l}.crop(3)+net2.layers{l}.crop(4)];
        dydz_sz(1:2) = net2.layers{l}.upsample.*(dydz_sz(1:2) - 1) + f_sz(1:2) - crops;
        fprintf('  out -> %4.0f %4.0f %4.0f\n', dydz_sz(1), dydz_sz(2), weights_sz(3));
    end
end
disp(' ');

end

