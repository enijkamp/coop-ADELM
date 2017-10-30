function [ind,z] = fake_fn(n,alphas,target)
    ind = linspace(-1,12,n);
    alphas = [0,alphas];
    z = zeros(length(alphas),n);
    for a = 1:length(alphas)
        for i = 1:n
            z(a,i) = f(ind(i))+penalty(ind(i),alphas(a),target);
        end
    end
    cyans = linspace(0,.6,length(alphas));
    plot(ind,z(1,:),'Color',[0,0,0]);
    hold on;
    for i = 2:length(alphas)
        plot(ind,z(i,:),'Color',[0,cyans(i),cyans(i)]);
    end
    xlim([-1,12]);
    ylim([-0.05,.85]);
    leg=legend('\alpha = 0 (original landscape)','\alpha = 0.015','\alpha = 0.03','\alpha = 0.05','\alpha = 0.08','Location','northeast');
    xlab = xlabel('Position');
    set(xlab,'FontSize',18);
    ylab = ylabel('Energy');
    set(ylab,'FontSize',18);
    titl = title(['Magnetized Landscape, Target Position = ',num2str(target)]);
    set(titl,'FontSize',18);
    set(gca,'FontSize',15);
    set(leg,'FontSize',12);
    hold off;
end

function y = f(x)
    y =x^2*(x<0) +(x>4)*(x<4.5)*((x-4))^2+ (x>4.5)*(x<5)*((x-5))^2 + (x>11)*(x-11)^4+ .002*sin(10*x) + .003*sin(20*x)+.0085*cos(3*x)+0.004*sin(25*x);
end

function p = penalty(x,alpha,target)
    p = alpha * abs(x-target);
end