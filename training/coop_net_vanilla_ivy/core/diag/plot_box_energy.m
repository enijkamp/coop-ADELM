function plot_box_energy( out_dir, neg, config, des_net, multi_grid )

%% (1) all exps

% paths
exps = {};

% train
exps{end+1} = { '../data/ivy/all/', '../data/ivy/all/*.png', 'train' };

% old - mean substraction
exps{end+1} = { '../ims_syn/ivy/all_4_1/', '../ims_syn/ivy/all_4_1/*.png', 'all_4_1 1' };
exps{end+1} = { '../ims_gen/ivy/all_4_1/', '../ims_gen/ivy/all_4_1/*.png', 'all_4_1 2' };

% new - ivy
dirs = dir('figure/ivy*');
for i = 1:length(dirs)
    exp_path = [ 'figure/' dirs(i).name ];
    if ~isempty(dir([exp_path '/net1*.png'])) && ~isempty(dir([exp_path '/net2*.png']))
        name_parts = strsplit(dirs(i).name,'_');
        name = [name_parts{end-1} '_' name_parts{end}];
        exps{end+1} = { exp_path, [exp_path '/net1*.png'], [name ' 1'] };
        exps{end+1} = { exp_path, [exp_path '/net2*.png'], [name ' 2'] };
    end
end

% energy
energies = {};
groups = {};
for i = 1:length(exps)
    exp_path = exps{i};
    if multi_grid
        images = read_images_raw(exp_path{1}, dir(exp_path{2}), [64, 64, 3]);
    else
        images = read_images(exp_path{1}, dir(exp_path{2}), [64, 64, 3]);
    end
    energies{i} = compute_energy(config,des_net,images,multi_grid);
    groups = horzcat(groups, repmat({exp_path{3}},1,length(energies{i})));
end

% box
f1 = figure('pos',[10 10 2000 1000]);
hold on;
boxplot(flat(energies), groups);
xtickangle(90);
set(gca,'fontsize',8);
set(gca, 'LooseInset', get(gca,'TightInset'));
hold off;
boxes = findobj(gcf,'tag','Box');
set(boxes(1:2:end), 'Color', [0 0 0]);

% mark non-realistic negatives
indices = [];
for i = 1:length(exps)
    exp_path = exps{i};
    if any(contains(exp_path(1), neg))
        indices(end+1) = i;
    end
end
for i = 1:length(indices)
    ind = indices(i);
    set(boxes(end-ind+1), 'Color', [1 0 0]);
end

% save
mkdir(out_dir);
save([out_dir 'energy.mat'], 'energies', 'groups');
saveas(f1, [out_dir 'energy_box.fig']);
saveas(f1, [out_dir 'energy_box.png']);
saveas(f1, [out_dir 'energy_box.pdf']);

%% (2) good exps

% paths
exps = {};

% train
exps{end+1} = { '../data/ivy/all/', '../data/ivy/all/*.png', 'train' };

% old - mean substraction
exps{end+1} = { '../ims_syn/ivy/all_4_1/', '../ims_syn/ivy/all_4_1/*.png', 'all_4_1 1' };
exps{end+1} = { '../ims_gen/ivy/all_4_1/', '../ims_gen/ivy/all_4_1/*.png', 'all_4_1 2' };

% new - ivy
dirs = dir('figure/ivy*');
for i = 1:length(dirs)
    exp_path = [ 'figure/' dirs(i).name ];
    if ~any(contains(exp_path, neg))
        if ~isempty(dir([exp_path '/net1*.png'])) && ~isempty(dir([exp_path '/net2*.png']))
            name_parts = strsplit(dirs(i).name,'_');
            name = [name_parts{end-1} '_' name_parts{end}];
            exps{end+1} = { exp_path, [exp_path '/net1*.png'], [name ' 1'] };
            exps{end+1} = { exp_path, [exp_path '/net2*.png'], [name ' 2'] };
        end
    end
end

% energy
energies = {};
groups = {};
for i = 1:length(exps)
    exp_path = exps{i};
    if multi_grid
        images = read_images_raw(exp_path{1}, dir(exp_path{2}), [64, 64, 3]);
    else
        images = read_images(exp_path{1}, dir(exp_path{2}), [64, 64, 3]);
    end
    energies{i} = compute_energy(config,des_net,images,multi_grid);
    groups = horzcat(groups, repmat({exp_path{3}},1,length(energies{i})));
end

% box
f1 = figure('pos',[10 10 2000 1000]);
hold on;
boxplot(flat(energies), groups);
xtickangle(90);
set(gca,'fontsize',8);
set(gca, 'LooseInset', get(gca,'TightInset'));
hold off;
boxes = findobj(gcf,'tag','Box');
set(boxes(1:2:end), 'Color', [0 0 0]);

% save
mkdir(out_dir);
save([out_dir 'energy_good.mat'], 'energies', 'groups');
saveas(f1, [out_dir 'energy_box_good.fig']);
saveas(f1, [out_dir 'energy_box_good.png']);
saveas(f1, [out_dir 'energy_box_good.pdf']);

end


function out = flat(energies)
out = [];
for i = 1:length(energies)
    en = energies{i};
    out = horzcat(out, en');
end
end

function en = compute_energy(config,net1,images,multi_grid)
en = zeros(size(images,4), 1);
for i = 1:size(images,4)
    if multi_grid
        en(i) = get_im_energy_multigrid(config,net1,images(:,:,:,i));
    else
        en(i) = get_im_energy(config,net1,images(:,:,:,i));
    end
end
end

function [img_mat] = read_images(inPath, files, imageSize)

if isempty(files)
    fprintf(['error: No training images are found in "' inPath '"\n']);
    keyboard;
end

img_mat = zeros([imageSize, length(files)], 'single');
for iImg = 1:length(files)
    fprintf('read and process images %d / %d\n', iImg, length(files))
    img = single(imread(fullfile(inPath, files(iImg).name)));
    img = imresize(img, imageSize(1:2));
    min_val = min(img(:));
    max_val = max(img(:));
    img_mat(:,:,:,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
end

end

function [img_mat] = read_images_raw(inPath, files, imageSize)

if isempty(files)
    fprintf(['error: No training images are found in "' inPath '"\n']);
    keyboard;
end

img_mat = zeros([imageSize, length(files)], 'single');
for iImg = 1:length(files)
    fprintf('read and process images %d / %d\n', iImg, length(files))
    img = single(imread(fullfile(inPath, files(iImg).name)));
    img_mat(:,:,:,iImg) = imresize(img, imageSize(1:2));
end

end
