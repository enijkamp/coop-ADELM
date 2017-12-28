function [ dirs ] = filter_dirs( exp_id, run_ids )

dirs = {};
for i = 1:length(run_ids)
    dirs{i} = [ 'ivy_dense_em_' num2str(exp_id) '_' num2str(run_ids(i)) ];
end

end

