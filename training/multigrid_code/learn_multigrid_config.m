function [net1, net2, net3, config] = learn_multigrid_config(config)

learningTime = tic;

%% Setup Network 1
net1 = convNet_object_1to4(config.res(2));
net2 = convNet_object_4to16(config.res(3));
net3 = convNet_object_16to64(config.res(4));

net1 = setup_network(net1);
net2 = setup_network(net2);
net3 = setup_network(net3);
%% Step 2 create imdb
[imdb, getBatch, net3] = create_imdb(config, net3);
net2.normalization.averageImage = ones(net2.normalization.imageSize) ...
        * net3.normalization.averageImage(1,1,1,1);
net1.normalization.averageImage = ones(net1.normalization.imageSize) ...
        * net3.normalization.averageImage(1,1,1,1);

%% Step 3: training
[net1, net2, net3, config] = train_model_multigrid(config, net1, net2, net3, imdb, getBatch);

learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);