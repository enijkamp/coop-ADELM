function [] = exp_texture_joint_20()

%% exp
exp_num = 20;

%% (1)

% (20): refine (9_1), determine burn-in, not substracting mean-image

% result:
%   1   -> uniform light green
%   2   -> some structure
%   3-5 -> almost uniform dark green

% %% config
% nIteration = 60;
% batchSize = 50;
% num_syn = 120;
% substract_mean = false;
% 
% % descriptor
% Deltas = 0.3;
% Gamma = 0.00005;
% Decay = [10*ones(1,100), 5*ones(1,100), 1.0*ones(1,100), 0.5*ones(1,100), 0.1*ones(1,100), 0.05*ones(1,100)];
% Ts = 10;
% 
% % generator
% Delta2 = 0.3; % unused
% Gammas2 = [0.0002*10 0.0002*5 0.0002*1 0.0002*0.5 0.0002*0.1];
% Decay2 = [10*ones(1,100), 5*ones(1,100), 1.0*ones(1,100), 0.5*ones(1,100), 0.1*ones(1,100), 0.05*ones(1,100)];

%% (2)

% (20): fix gamma2, increase gamma1

% result:
%   none of them work, degenerate into green uniform

% %% config
% nIteration = 500;
% batchSize = 50;
% num_syn = 120;
% substract_mean = false;
% 
% % descriptor
% Deltas = 0.3;
% Gammas = [0.1 0.05 0.01];
% Decay = linspace(1.0, 0.001, nIteration);
% Ts = 10;
% 
% % generator
% Delta2 = 0.3; % unused
% Gammas2 = [0.01 0.001 0.0001];
% Decay2 = linspace(1.0, 0.001, nIteration);
% 
% % image
% img_name = 'ivy';
% img_size = 'all';
% 
% %% setup
% rng(123);
% use_gpu = 1;
% compile_convnet = 1;
% compile_cudnn = 1;

%% (3)

% (20): logspace decay

% result:
%   ...

%% config
nIteration = 100;
batchSize = 50;
num_syn = 120;
substract_mean = false;
rate_list = logspace(-2, -4, 80)*100;

% descriptor
Ts = 10;
Deltas = [0.3 0.005];
Gammas = [0.1 0.06 0.04 0.02 0.01];
Decay = repmat(rate_list, max(1,floor(nIteration/length(rate_list))),1);
Decay = reshape(Decay, 1, []);
Decay(end:nIteration) = Decay(end);


% generator
Delta2 = 0.3; % unused
Gammas2 = [0.001 0.0001 0.00001 0.000001];
Decay2 = repmat(rate_list, max(1,floor(nIteration/length(rate_list))),1);
Decay2 = reshape(Decay2, 1, []);
Decay2(end:nIteration) = Decay2(end);

% image
img_name = 'ivy';
img_size = 'all';

%% setup
rng(123);
use_gpu = 1;
compile_convnet = 1;
compile_cudnn = 1;

setup_path();
setup_convnet(use_gpu, compile_convnet, compile_cudnn);
disp(gpuDevice());

%% run
num = 1;
total = length(Deltas) * length(Gammas2) * length(Ts);
for Delta = Deltas
    for Gamma = Gammas
        for Gamma2 = Gammas2
            for T = Ts
                % log
                disp(['### run ' num2str(num) ' / ' num2str(total) ' ###']);

                % images
                prefix = [img_name '/' num2str(img_size) '_' num2str(exp_num) '_' num2str(num) '/'];
                [config, net1] = train_coop_config();
                config = prep_dirs(config, prefix);
                config = load_images(config);
                config.use_gpu = use_gpu;

                % config
                config.nIteration = nIteration;
                config.batchSize = batchSize;
                config.substract_mean = substract_mean;

                % sampling parameters
                config.num_syn = num_syn;

                % descriptor net1 parameters
                config.Delta = Delta;
                config.Gamma = Decay * Gamma;
                config.refsig = 1;                                            
                config.T = T;            

                % generator net2 parameters
                config.Delta2 = Delta2;
                config.Gamma2 = Decay2 * Gamma2;
                config.refsig2 = 1;
                config.s = 0.3;
                config.real_ref = 1;
                config.cap2 = 8;

                % learn
                learn_dual_net(config, net1);

                num = num + 1;
            end
        end
    end
end
end
