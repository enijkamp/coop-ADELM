file_str = '/Users/mitch/Dropbox/Coop ADELM/maps/ivy/';
strs{1} = '128/ELM_exp'; strs{2} = '512/ELM_exp'; strs{3} = '2048/ELM_exp';
%alpha_seq = zeros(3,10);
%num_mins = zeros(3,10);

if 0
for i = 1:3
    for j = 1:10
        load([file_str,strs{i},num2str(j),'.mat']);
        alpha_seq(i,j) = ELM.config.alpha;
        num_mins(i,j) = length(ELM.min_IDs);
    end
end
end

num_mins(1,1) = 1241;
num_mins(3,1) = 1400;
plot(log(alpha_seq(1,2:10)),log(num_mins(1,2:10)));
hold on;
plot(log(alpha_seq(2,2:10)),log(num_mins(2,2:10)));
plot(log(alpha_seq(3,2:10)),log(num_mins(3,2:10)));
num_mins(1,4)=7;
xvals = linspace(4,12,1000);
X1 = [ones(3,1),log(alpha_seq(1,2:4))'];
Y1 = log(num_mins(1,2:4))';
b1 = X1\Y1;
d1 = b1(1) + xvals*b1(2);

X2 = [ones(9,1),log(alpha_seq(2,2:10))'];
Y2 = log(num_mins(2,2:10))';
b2 = X2\Y2;
d2 = b2(1) + xvals*b2(2);

X3 = [ones(9,1),log(alpha_seq(3,2:10))'];
Y3 = log(num_mins(3,2:10))';
b3 = X3\Y3;
d3 = b3(1) + xvals*b3(2);

plot(xvals,d1,'--','Color',[0,0.4470,0.7410]);
plot(xvals,d2,'--','Color',[0.8500,0.3250,0.0980]);
plot(xvals,d3,'--','Color',[0.9290,0.6940,0.125]);

xlabel('log(alpha)');
ylabel('log(Num. Min Found)');
title('Multi-Resolution ADELM for 3 Scales of Ivy');
legend('Texture','Texton','Basis','Location','best');
xlim([5.5,10]);
ylim([0,7]);
text(6.3,4,'slope = -3.85','FontSize',14);
text(7.2,3,'slope = -2.18','FontSize',14);
text(8.2,4,'slope = -2.06','FontSize',14);
hold off;
figure(1);