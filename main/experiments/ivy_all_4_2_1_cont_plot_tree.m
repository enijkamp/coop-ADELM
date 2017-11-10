function [] = ivy_all_4_2_1_plot_tree()

exp_id = '4_1_2_cont';
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

% remove deep modes

nodes_list = [ nodes{:} ];
energy_flat = [nodes_list(:).energy];
[~, en_ind] = sort(energy_flat);

for i = 1:10
    ELM.min_ims(:,:,:,en_ind(i)) = ELM.min_ims(:,:,:,en_ind(11));
    nodes{en_ind(i)}.energy = nodes{en_ind(11)}.energy;
end


[min_viz_ord,viz_mat_x,viz_mat_y] = viz_tree(nodes,ELM,'',0,0.03,.5*10^5);

saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '_2.eps'],'eps2c');
saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '_2.png'],'png');

end

