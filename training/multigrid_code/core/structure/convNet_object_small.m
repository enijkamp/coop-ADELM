function net = convNet_object(config)
net.layers = [];
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = true;
opts.addrelu = true;


%% layer 1
layer_name = '1';
num_in = 3;
num_out = 96;
filter_sz = 5;
stride = 2;
pad = 2 * ones(1, 4);
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);

%% layer2
layer_name = '2';
num_in = num_out;
num_out = 256;
filter_sz = 5; 
stride = 2;
pad = 2 * ones(1, 4);
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;


img = randn([config.sx, config.sy, 3], 'single');
net = vl_simplenn_move(net, 'gpu') ;
res = vl_simplenn(net, gpuArray(img));
net = vl_simplenn_move(net, 'cpu');
dydz_sz = size(res(end).x);

                                             
%% top layer
opts.batchNormalization = false;
numFilters = 10; %% 
stride = 1;
pad_sz = 0;
pad = ones(1,4)*pad_sz;

opts.addrelu = false;

layer_name = num2str(str2num(layer_name)+1);
net = add_cnn_block(net, opts, layer_name, dydz_sz(1), dydz_sz(1), num_out, numFilters, stride, pad, 0.1);


