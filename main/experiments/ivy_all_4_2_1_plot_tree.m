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

% substitute deep mode with it's left neighbour

min_e = inf;
min_i = 0;
for i = 1:length(min_viz_ord)
    if nodes{i}.energy < min_e
        min_e = nodes{i}.energy;
        min_i = i;
    end
end

min_viz_ord = get_viz_order(nodes);
order_min_i = find(min_viz_ord == min_i);
min_1_neighbour = min_viz_ord(order_min_i - 1);

disp(min_viz_ord);
disp(num2str(min_i));
disp(num2str(min_1_neighbour));

ELM.min_ims(:,:,:,min_i) = ELM.min_ims(:,:,:,41);
nodes{min_i}.energy = nodes{41}.energy;

[min_viz_ord,viz_mat_x,viz_mat_y] = viz_tree(nodes,ELM,'',0,0.03);

saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '_2.eps'],'eps2c');
saveas(gcf,['../../plots/trees/' file_str '/tree_' exp_id '_2.png'],'png');

end

function [min_viz_order] = get_viz_order(nodes)
    ind = length(nodes);
    temp_order = ind;
    while ind > 0 && ~isempty(nodes{ind}.children)
        exp_ind = find(temp_order == ind);
        temp_order=[temp_order(1:(exp_ind-1)),nodes{ind}.children, ...
                        temp_order((exp_ind+1):length(temp_order)) ];
        ind = ind - 1;
    end
    min_viz_order =temp_order;
end

