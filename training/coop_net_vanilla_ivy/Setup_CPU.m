function [] = Setup_CPU(compile)

current_dir = pwd();

cd(fullfile('../../matconvnet-1.0-beta16/', 'matlab'));
cuda_root = '/usr/local/cuda-8.0'; 
cudnn_root = '/home/enijkamp/cudnn-3.0'; 

vl_setupnn();

if compile
vl_compilenn('EnableGPU', false);
end

% vl_compilenn('EnableGPU', true, ...
%     'CudaRoot', cuda_root, ...
%     'CudaMethod', 'nvcc', ...
%     'enableCudnn', false);


cd(current_dir);