function [] = setup_convnet(use_gpu, do_compile, use_cudnn)

if nargin < 3
    use_cudnn = 0;
end

current_dir = pwd();

if use_gpu
    % gpu
    cd(fullfile('../../matconvnet-1.0-beta16-gpu/', 'matlab'));
    cuda_root = '/usr/local/cuda-8.0'; 
    cudnn_root = '/home/enijkamp/cudnn-3.0'; 
    
    vl_setupnn();
    
    if do_compile
        if use_cudnn
            vl_compilenn('EnableGPU', true, ...
                'CudaRoot', '/usr/local/cuda-8.0';, ...
                'CudaMethod', 'nvcc', ...
                'enableCudnn', true, ...
                'cudnnRoot', '/opt/cudnn-3.0');
        else
            vl_compilenn('EnableGPU', true, ...
                'CudaRoot', cuda_root, ...
                'CudaMethod', 'nvcc');
        end
    end
else
    % cpu
    cd(fullfile('../../matconvnet-1.0-beta16/', 'matlab'));
    vl_setupnn();
    if do_compile
        vl_compilenn();
    end
end

cd(current_dir);

end
