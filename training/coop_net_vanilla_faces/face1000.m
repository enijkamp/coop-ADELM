src_dir = '../../data/celebA/';
dst_dir = '../../data/face10000/';

img_list = dir([src_dir, '*.jpg']);

if ~exist(dst_dir, 'dir')
    mkdir(dst_dir);
end

idx = numel(img_list);
idx = randperm(idx, 12000);

for i = idx
    cmd = sprintf('cp %s%s %s%s', src_dir, img_list(i).name, dst_dir, img_list(i).name);
    system(cmd);
end