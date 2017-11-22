function [] = experiment_diag_ivy_1_old()

%% prep
clear all;
close all;
restoredefaultpath();
addpath(genpath('./core/'));
Setup_CPU(false);
rng(123);

%% load
load('working/ivy_dense_em_0_1/layer_01_iter_2200_model.mat');
load('working/ivy_dense_em_0_1/config.mat');

if 0
images_train = read_images('../data/ivy/all/', dir(['../data/ivy/all/' '*.png']), [64, 64, 3]);
en1 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    disp(num2str(i));
    im = images_train(:,:,:,i);
    en1(i) = get_im_energy(config,net1,im);
end
save('en1.mat', 'en1');
end
load('en1.mat', 'en1');

images_train = read_images('figure/ivy_dense_em_10_2/', dir('figure/ivy_dense_em_10_2/net1*.png'), [64, 64, 3]);
en2_1 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en2_1(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

images_train = read_images('figure/ivy_dense_em_10_2/', dir('figure/ivy_dense_em_10_2/net2*.png'), [64, 64, 3]);
en2_2 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en2_2(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

images_train = read_images('figure/ivy_dense_em_11_5/', dir('figure/ivy_dense_em_11_5/net1*.png'), [64, 64, 3]);
en3_1 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en3_1(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

images_train = read_images('figure/ivy_dense_em_11_5/', dir('figure/ivy_dense_em_11_5/net2*.png'), [64, 64, 3]);
en3_2 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en3_2(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

images_train = read_images('figure/ivy_dense_em_11_7/', dir('figure/ivy_dense_em_11_7/net1*.png'), [64, 64, 3]);
en4_1 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en4_1(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

images_train = read_images('figure/ivy_dense_em_11_7/', dir('figure/ivy_dense_em_11_7/net2*.png'), [64, 64, 3]);
en4_2 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en4_2(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

images_train = read_images('../ims_syn/ivy/all_4_1/', dir('../ims_syn/ivy/all_4_1/*.png'), [64, 64, 3]);
en5_1 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en5_1(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

images_train = read_images('../ims_gen/ivy/all_4_1/', dir('../ims_gen/ivy/all_4_1/*.png'), [64, 64, 3]);
en5_2 = zeros(size(images_train,4), 1);
for i = 1:size(images_train,4)
    en5_2(i) = get_im_energy(config,net1,images_train(:,:,:,i));
end

% hist
[counts1, binCenters1] = hist(en1, 50);
[counts2_1, binCenters2_1] = hist(en2_1, 50);
[counts2_2, binCenters2_2] = hist(en2_2, 50);
[counts3_1, binCenters3_1] = hist(en3_1, 50);
[counts3_2, binCenters3_2] = hist(en3_2, 50);
[counts4_1, binCenters4_1] = hist(en4_1, 50);
[counts4_2, binCenters4_2] = hist(en4_2, 50);
f1 = figure;
plot(binCenters1, counts1, 'r-');
hold on;
plot(binCenters2_1, counts2_1, 'g-');
plot(binCenters2_2, counts2_2, 'b-');
plot(binCenters3_1, counts3_1, 'y-');
plot(binCenters3_2, counts3_2, 'm-');
plot(binCenters4_1, counts4_1, 'c-');
plot(binCenters4_2, counts4_2, 'k-');
grid on;
hold off;
legend({'train', '10 2 net1', '10 2 net2', '11 5 net1', '11 5 net2', '11 7 net1', '11 7 net2'});

% box
f2 = figure;
hold on;
grp = horzcat(repmat({'train'},1,length(en1)), ...
    repmat({'10 2 net1'},1,length(en2_1)), ...
    repmat({'10 2 net2'},1,length(en2_2)), ...
    repmat({'11 5 net1'},1,length(en3_1)), ...
    repmat({'11 5 net2'},1,length(en3_2)), ...
    repmat({'11 7 net1'},1,length(en4_1)), ...
    repmat({'11 7 net2'},1,length(en4_2)), ...
    repmat({'all 4 1 net1'},1,length(en5_1)), ...
    repmat({'all 4 1 net2'},1,length(en5_2)));
boxplot([en1' en2_1' en2_2' en3_1' en3_2' en4_1' en4_2' en5_1' en5_2'], grp);
hold off;

saveas(f1, 'energy_hist.png');
saveas(f2, 'energy_box.png');

end