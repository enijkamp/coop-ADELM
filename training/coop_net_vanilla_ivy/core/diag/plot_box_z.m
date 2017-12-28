function plot_box_z( dirs, out_dir, neg )

%% (1) all exps

% exps
exps = {};
for i = 1:length(dirs)
    exp_path = [ dirs(i).folder '/' dirs(i).name ];
    if ~isempty(dir(exp_path))
        name_parts = strsplit(dirs(i).name,'_');
        name = [name_parts{end-1} '_' name_parts{end}];
        exps{end+1} = { exp_path, name, name };
    end
end

% l2 norm
dists = {};
groups = {};
for i = 1:length(exps)
    exp_path = exps{i};
    exp_loaded = load([exp_path{1} '/images_min_dist.mat' ], 'images_min_dist');
    dists{i} = exp_loaded.images_min_dist;
    groups = horzcat(groups, repmat({exp_path{3}},1,length(dists{i})));
end

% box
f1 = figure('pos',[10 10 2000 1000]);
drawnow;
hold on;
boxplot(flat(dists), groups);
xtickangle(90);
set(gca,'fontsize',8);
set(gca, 'LooseInset', get(gca,'TightInset'));
hold off;

% mark non-realistic negatives
boxes = findobj(gcf,'tag','Box');
indices = [];
for i = 1:length(exps)
    exp_path = exps{i};
    if any(contains(exp_path{1}, neg))
        indices(end+1) = i;
    end
end
for i = 1:length(indices)
    ind = indices(i);
    set(boxes(end-ind+1), 'Color', [1 0 0]);
end

% save
mkdir(out_dir);
save([out_dir 'dists.mat'], 'dists', 'groups');
saveas(f1, [out_dir 'dists_box.fig']);
saveas(f1, [out_dir 'dists_box.png']);
saveas(f1, [out_dir 'dists_box.pdf']);

%% (2) good exps

% exps
exps = {};
for i = 1:length(dirs)
    exp_path = [ dirs(i).folder '/' dirs(i).name ];
    if ~any(contains(exp_path, neg)) && ~isempty(dir(exp_path))
        name_parts = strsplit(dirs(i).name,'_');
        name = [name_parts{end-1} '_' name_parts{end}];
        exps{end+1} = { exp_path, name, name };
    end
end

% l2 norm
dists = {};
groups = {};
for i = 1:length(exps)
    exp_path = exps{i};    
    exp_loaded = load([exp_path{1} '/images_min_dist.mat' ], 'images_min_dist');
    dists{i} = exp_loaded.images_min_dist;
    groups = horzcat(groups, repmat({exp_path{3}},1,length(dists{i})));
end

% box
f2 = figure('pos',[10 10 2000 1000]);
drawnow;
hold on;
boxplot(flat(dists), groups);
xtickangle(90);
set(gca,'fontsize',8);
set(gca, 'LooseInset', get(gca,'TightInset'));
hold off;

% save
mkdir(out_dir);
save([out_dir 'dists_good.mat'], 'dists', 'groups');
saveas(f2, [out_dir 'dists_box_good.fig']);
saveas(f2, [out_dir 'dists_box_good.png']);
saveas(f2, [out_dir 'dists_box_good.pdf']);

end

function out = flat(energies)
out = [];
for i = 1:length(energies)
    en = energies{i};
    out = horzcat(out, en');
end
end
