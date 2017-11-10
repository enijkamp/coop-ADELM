function [ind,z,z2] = fake_fn2(type)
if type == 1
    n=10000;
    ind = linspace(.001,1,n);
    z = zeros(1,n);
    z2 = zeros(1,n);
    for i = 1:n
        z(i) = f(ind(i));
        z2(i) = -z(i);
    end
    plot(ind,z,'--','Color',[0,.447,.741],'LineWidth',1.3);line([0,1],[0,0],'Color','r','LineWidth',1.3);hold on; scatter(1,0,[],[0,.447,.741],'filled');hold off;ylim([-10,10]);xlim([-.1,1.2]);set(gca,'XTick',[]);set(gca,'YTick',[]);legend('Metastable Boundary','First-Order Transition Boundary','Critical Temperature');title('Phase Diagram of Magnetized Ising Hamiltonian');xlabel('Temperature (T)');ylabel('Magnetic Field Strength (H)');line(ind,z2,'Color',[0,.447,.741],'LineStyle','--','LineWidth',1.3);line([0,2],[0,0],'Color',[0,0,0]);line([0,0],[-100,100],'Color',[0,0,0]);line([0,1],[0,0],'Color','r','LineWidth',1.5);hold on; scatter(1,0,[],[0,.447,.741],'filled');hold off;figure(1);
    set(gca,'FontSize',15);
elseif type ==2
    nada;
elseif type ==3
    nada2;
else 
    nada3;
end


end

function y = f(x)
    %y =x^2*(x<0) +(x>4)*(x<4.5)*((x-4))^2+ (x>4.5)*(x<5)*((x-5))^2 + (x>11)*(x-11)^4+ .002*sin(10*x) + .003*sin(20*x)+.0085*cos(3*x)+0.004*sin(25*x);
    y = abs(x-1)/x^.5;
end

function p = penalty(x,alpha)
    p = alpha * abs(x-10);
end

function nada
ind = linspace(0,5,5000);
z1 = sqrt(ind(1:1000));
z2 = 1./(1+exp(-1*ind));
z3 = [z1,z2*2];
plot(z3);figure(1);
z4 = -wrev(z3);
plot(z4);figure(1);
ind = linspace(0,6,6000);
revind = -wrev(ind);
z4 = z4/2-.5;
z3 = z3/2+.5;

plot(revind(1:5000)+1,z4(1:5000),'Color',[0,.447,.741]);
line(revind(5001:end)+1,z4(5001:end),'LineStyle', '--','Color',[0,.447,.741]);
line([-1,-1],[-.05,.05],'Color','b','LineWidth',1.5);
line([-100,100],[1.5,1.5],'Color','r','LineStyle','--');
legend('Stable State','Metastable State','Metastable Interval','Perfect Spin Alignment','Location','northwest');
line(ind(1001:end)-1,z3(1001:end),'Color',[0,.447,.741]);
hold on; scatter([-1,1],[1/2,-1/2],10,[0,.447,.741],'filled');hold off;
line(ind(1:1000)-1,z3(1:1000),'LineStyle','--','Color',[0,.447,.741]);
xlim([-3,3]);ylim([-1.75,1.75]);
line([-100,100],[0,0],'Color',[0,0,0]);
line([0,0],[-100,100],'Color',[0,0,0]);
line([-100,100],[-1.5,-1.5],'Color','r','LineStyle','--');
line([1,1],[-.05,.05],'Color','b','LineWidth',1.5);
line([-1,1],[0,0],'Color','b','LineWidth',1.5);
set(gca,'XTick',[]);set(gca,'YTick',[]);
title('Magnetization of Ising Model Below Critical Temperature');
xlabel('Magnetic Field Strength (H)');
ylabel('Magnetization of System (M)');
figure(1);
end

function nada2
    n=10000;
    ind = linspace(.001,1,n);
    ind2 = linspace(.001,.65,n);
    z = zeros(1,n);
    z2 = zeros(1,n);
    for i = 1:n
        z(i) = (f(ind(i))/2)^2;
        z2(i) = z(i);
    end
    
    scatter3(1,0,0,[],[0,.447,.741],'filled');
    hold on;
    scatter3(.65,0,0,[],[0.8500,0.3250,0.0980],'filled');
    plot3([1,1],[0,0],[0,0],'Color',[0,.447,.741]);
    plot3([.65,.65],[0,0],[0,0],'Color',[0.8500,0.3250,0.0980]);
    legend('Min 1 Critical Temperature','Min 2 Critical Temperature','Min 1 Metastable Region','Min 2 Metastable Region');
    plot3(ind,zeros(1,n),z,'LineStyle','--','LineWidth',1.5,'Color',[0,.447,.741]);
    plot3(ind2,z2,zeros(1,n),'LineStyle','--','LineWidth',1.5,'Color',[0.8500,0.3250,0.0980]);
    plot3([0,100],[0,0],[0,0],'Color',[0,0,0]);
    plot3([0,0],[0,100],[0,0],'Color',[0,0,0]);
    plot3([0,0],[0,0],[0,100],'Color',[0,0,0]);
    for i = 1:n
        if mod(i,100) ==0
            plot3([ind(i),ind(i)],[0,0],[0,z(i)],'Color',[0,.447,.741]);
        end
        
        if mod(i,300) ==0
            plot3([ind2(i),ind2(i)],[0,z2(i)],[0,0],'Color',[0.8500,0.3250,0.0980]);
        end
    end
    scatter3(1,0,0,[],[0,.447,.741],'filled');
    scatter3(.65,0,0,[],[0.8500,0.3250,0.0980],'filled');
    hold off;
    xlim([-.1,2]);ylim([-.1,2]);zlim([-.1,2]);
    set(gca,'XTick',[]);set(gca,'YTick',[]);set(gca,'ZTick',[]);
    title('Metastability of Two Minima under Attraction Diffusion');
    xlabel('Temperature (T)');
    ylabel('Magnetization towards Min 1');
    zlabel('Magnetization towards Min 2');
    set(gca,'FontSize',15);
    figure(1);
end


%[ind,z,z2]=fake_fn2(10000,1);plot(ind,z,'--b');line([0,1],[0,0],'Color','r');hold on; scatter(1,0,'filled');hold off;ylim([-10,10]);xlim([-.1,1.2]);set(gca,'XTick',[]);set(gca,'YTick',[]);legend('Metastable Boundary','First-Order Transition Boundary','Critical Temperature');title('Phase Diagram of Magnetized Ising Hamiltonian');xlabel('Temperature T');ylabel('Magnetization H');line(ind,z2,'Color','b','LineStyle','--');line([-2,2],[0,0],'Color',[0,0,0]);line([0,0],[-100,100],'Color',[0,0,0]);line([0,1],[0,0],'Color','r');figure(1);