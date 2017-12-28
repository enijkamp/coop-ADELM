function [] = Setup(cudnn, compile)

if nargin < 1
   cudnn = 1; 
   compile = 1;
end

current_dir = pwd();

cd(fullfile('../../matconvnet-1.0-beta16-gpu/', 'matlab'));
cuda_root = '/usr/local/cuda-8.0'; 
cudnn_root = '/home/enijkamp/cudnn-3.0'; 

vl_setupnn();

if cudnn

    if compile
        vl_compilenn('EnableGPU', true, ...
            'CudaMethod', 'nvcc', ...
            'CudaRoot', cuda_root, ...
            'enableCudnn', true, ...
            'cudnnRoot', cudnn_root);
    end

else

    if compile
        vl_compilenn('EnableGPU', true, ...
            'CudaMethod', 'nvcc', ...
            'CudaRoot', cuda_root, ...
            'enableCudnn', false);
    end

end

cd(current_dir);