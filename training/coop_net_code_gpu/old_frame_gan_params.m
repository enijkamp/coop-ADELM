function net = old_frame_gan_params()

net.layers = [];
opts.scale = 2 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = true;
opts.addrelu = false;
opts.type = 'convt';
opts.leak = 0.2;

%% layer 1
layer_name = '1';
num_in = 30;
num_out = 500;
filter_sz = 5; 
%filer_sz = 4; for 48x48
%filter_sz = 4; %for 32x32
upsample = 3;
%upsample = 3 %for 32x32
%upsample = 3;%for 48x48
crop= [0,0,0,0];
net = add_convt_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop);
net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',layer_name));

%% layer 2
layer_name = '2';
num_in = 500;
num_out = 100;
filter_sz = 7;
%filter_sz = 5; %for 32x32
%filter_sz = 6; for 48x48
upsample = 3;
%upsample = 3; %for 32x32
%upsample = 3; %for 48x48
crop = [1,2,1,2];
net = add_convt_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop);
net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',layer_name));

%% layer 4
layer_name = '3';
num_in = 100;
num_out = 3;
filter_sz = 7; %fully
% filter_sz = 5; %for 32x32
%filter_sz = 7; for 48x48
upsample = 4; %fully
%upsample = 3; %for 32x32
%upsample = 4; %for 48x48
crop = [1,2,1,2];%fully

net = add_convt_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop);
net.layers{end+1} = struct('type', 'tanh', 'name', sprintf('tanh%s',layer_name));

