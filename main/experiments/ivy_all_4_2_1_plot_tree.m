function [] = ivy_all_4_2_1_plot_tree()

exp_id = '4_1_2';
file_str = 'ivy/all/4_1';

addpath(genpath('../../main/'));

load(['../../trees/' file_str '/bar_mat_' exp_id '.mat'], 'bar_mat');
nodes = build_ultrametric_tree(bar_mat);

load(['../../maps/' file_str '/ELM_' exp_id '_exp.mat'], 'ELM');

if ~exist(['../../plots/trees/' file_str],'dir') mkdir(['../../plots/trees/' file_str]); end

% original

[min_viz_ord,viz_mat_x,viz_mat_y] = viz_tree(nodes,ELM,'',0,0.03);

saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '.eps'],'eps2c');
saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '.png'],'png');

% removed deep mode

ELM.min_ims(:,:,:,4) = ELM.min_ims(:,:,:,41);
nodes{4}.energy = nodes{41}.energy;

[min_viz_ord,viz_mat_x,viz_mat_y] = viz_tree(nodes,ELM,'',0,0.03);

saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '_2.eps'],'eps2c');
saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '_2.png'],'png');

end

