function [min_viz_ord,viz_mat_x,viz_mat_y] = viz_tree(nodes,ELM,title_str,num_ex,prop,offset)
    if nargin < 3, title_str = ''; end
    if nargin < 4, num_ex = 0; end
    if nargin < 5, prop = 0.03; end
    if nargin < 6, offset = 0; end
    %nodes = permute_children(nodes);
    min_viz_ord = get_viz_order(nodes);
    viz_mat_x = [];
    viz_mat_y = [];
    for i = 1:length(nodes)
        if isempty(nodes{i}.children)
            nodes{i}.x = find(min_viz_ord==i);           
        else
            child_inds = nodes{i}.children;
            child1 = nodes{child_inds(1)};
            child2 = nodes{child_inds(2)};
            nodes{i}.x = 0.5*(child1.x+child2.x);
            viz_mat_x(end+1,:) = [child1.x,child1.x,child2.x,child2.x];
            viz_mat_y(end+1,:) = [child1.energy,nodes{i}.energy, ...
                                    nodes{i}.energy,child2.energy];
        end
    end
    
    figure();
    set(gcf, 'Position', [10, 10, 1000, 600]);
    
    plot(viz_mat_x(1,:),viz_mat_y(1,:),'k');
    hold on;
    for i = 2:size(viz_mat_x,1)
        plot(viz_mat_x(i,:),viz_mat_y(i,:),'k');
    end
    min_e = flintmax;
    for i = 1:length(min_viz_ord)
        min_e = min([min_e,nodes{i}.energy]);
    end

    en_marg = (nodes{end}.energy-min_e);
    axis([0,length(min_viz_ord)+1,min_e-en_marg*prop*1.3,nodes{end}.energy+en_marg*prop/3]);
    % ticks
    xticks(1:length(min_viz_ord));
    xticklabels(string(1:length(min_viz_ord)));
    ax = gca;
    ax.XAxis.FontSize = 6;
    ax.YAxis.FontSize = 10;
    % axis
    xlabel('Minima Index','FontSize',11);
    ylabel('Energy','FontSize',11);
    
    title(title_str);  
    if nargin > 1 
        min_ims = ELM.min_ims;
%         for i = 1:size(min_ims,3)
%             diff = min_e-en_marg*.3*prop-(min_e-en_marg*prop);
%             inds = find(ismember(ELM.min_ID_path,min_viz_ord(i)));
%             pperm = randperm(length(inds));
%             inds = inds(pperm(1:min(num_ex,length(inds))));
%             ens = ELM.min_en_path(inds);
%             [~,ord]=sort(ens);
%             for j = 1:num_ex
%                 if length(ord)> num_ex-j
%                     temp_ind = num_ex-j+1;
%                     imagesc([i-7/16, i+7/16], [min_e-en_marg*(.3*prop)-(j-1)*diff, min_e-en_marg*prop-(j-1)*diff],ELM.min_locs(:,:,inds(ord(temp_ind))));
%                 end
%             end
%             imagesc([i-7/16, i+7/16], [min_e-en_marg*(.3*prop)-diff*(num_ex+1.25), min_e-en_marg*prop-diff*(num_ex+1.25)], ...
%                             min_ims(:,:,min_viz_ord(i)));
%             %text([i-7/16, i+7/16], [min_e-en_marg*(.3*prop)-diff*(num_ex+1.25), min_e-en_marg*prop-diff*(num_ex+1.25)],num2str(sum(ELM.min_ID_path==min_viz_ord(i))));
%             colormap('Gray');
%         end
        for i = 1:size(min_ims,4)
            diff = min_e-en_marg*.3*prop-(min_e-en_marg*prop);
            inds = find(ismember(ELM.min_ID_path,min_viz_ord(i)));
            pperm = randperm(length(inds));
            inds = inds(pperm(1:min(num_ex,length(inds))));
            ens = ELM.min_en_path(inds);
            [~,ord]=sort(ens);
%             for j = 1:num_ex
%                 if length(ord)> num_ex-j
%                     temp_ind = num_ex-j+1;
%                     imagesc([i-7/16, i+7/16], [min_e-en_marg*(.3*prop)-(j-1)*diff, min_e-en_marg*prop-(j-1)*diff],ELM.min_locs(:,:,inds(ord(temp_ind))));
%                 end
%             end

%             imagesc([i-7/16, i+7/16], [min_e-en_marg*(.3*prop)-diff*(num_ex+1.25), min_e-en_marg*prop-diff*(num_ex+1.25)], ...
%                             double(ELM.min_ims(:,:,:,min_viz_ord(i)))/256 );

            imagesc([i-7/16, i+7/16], [min_e-en_marg*(.3*prop)-diff*(num_ex+1.25)+offset, min_e-en_marg*prop-diff*(num_ex+1.25)+offset], ...
                            double(ELM.min_ims(:,:,:,min_viz_ord(i)))/256 );
                        
            %text([i-7/16, i+7/16], [min_e-en_marg*(.3*prop)-diff*(num_ex+1.25), min_e-en_marg*prop-diff*(num_ex+1.25)],num2str(sum(ELM.min_ID_path==min_viz_ord(i))));
        end
        %set(gca,'dataAspectRatio',[7/8,  diff, 1]);
        ylim([min_e-en_marg*prop-diff*(num_ex+2.25),-8*10^6]);
        %line([-1000,1000],[min_e-en_marg*(.3*prop)-diff*(num_ex), min_e-en_marg*prop-diff*num_ex],'LineStyle','--','Color',[0,0,0],'LineWidth',1.5);
    end
    %set(gca,'FontSize',12);
    hold off;
end

function nodes = permute_children(nodes)
    for i = 1:length(nodes)
        if ~isempty(nodes{i}.children)
            if rand < 0.5
                nodes{i}.children = [nodes{i}.children(2),nodes{i}.children(1)];
            end
        end
    end
    
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