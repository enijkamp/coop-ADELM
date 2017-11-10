function [] = exp_escher_1()

%% exp
exp_num = 1;

%% (1)

%% config
nIteration = 200;
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
img_name = 'escher';

%% setup
restoredefaultpath();
rng(123);
use_gpu = 1;
compile_cudnn = 1;
compile_convnet = 0;

setup_path();
setup_convnet(use_gpu, compile_convnet, compile_cudnn);

%% verify
disp(gpuDevice());
disp(parallel.internal.gpu.CUDADriverVersion);
disp(getenv('LD_LIBRARY_PATH'));
dev = gpuDevice();
assert(strcmp(dev.Name, 'TITAN X (Pascal)'));
assert(dev.ToolkitVersion == 8);
assert(contains(getenv('LD_LIBRARY_PATH'), 'cudnn-3.0'));

%% run
num = 1;
total = length(Deltas) * length(Gammas) * length(Gammas2) * length(Ts);
for Delta = Deltas
    for Gamma = Gammas
        for Gamma2 = Gammas2
            for T = Ts
                % log
                disp(['### run ' num2str(num) ' / ' num2str(total) ' ###']);

                % config
                prefix = [img_name '/'  num2str(exp_num) '_' num2str(num) '/'];
                [config] = train_coop_config();
                
                % override
                config.nIteration = nIteration;
                config.batchSize = batchSize;
                config.substract_mean = substract_mean;
                
                % load
                config = prep_dirs(config, prefix);
                config = load_images(config);
                config.use_gpu = use_gpu;

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
                learn_dual_net(config);

                num = num + 1;
            end
        end
    end
end
end

function root = setup_path()
root = '../../';
addpath([root 'training/coop_net_code_gpu_escher']);
end

