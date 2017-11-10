function [] = experiment_learn_dualNets_object()
close all;
category_path = '../Image/object/';
category_lists = dir([category_path, '*']);
valid_category = true(1, length(category_lists));
for i = 1:length(category_lists)
    if category_lists(i).name(1) == '.' || ~category_lists(i).isdir
       valid_category(i) = false; 
    end
end

category_lists = category_lists(valid_category);
category_lists = [];
category_lists(1).name = 'cliff_dwelling';
exp_type = 'object';

for ii = 1:length(category_lists)
    learn_dualNets(category_lists(ii).name, exp_type, 550);
end