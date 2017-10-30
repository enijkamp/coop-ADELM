function fake_fn3(net,config,meta_struc)
    meta24 = meta_struc.meta24;
    meta2loc = meta_struc.meta2loc;
    meta4loc = meta_struc.meta4loc;
    temps = meta_struc.temps;
    min2 = meta_struc.min2;
    min4 = meta_struc.min4;
    min_loc = meta_struc.min_loc;
    inds = 7:31;
    inds2 = 6:30;
    
    if 0
        meta2loc.o2(end-9) = exp(-2.75);
        plot(log(temps(inds)),log(meta24.o1(inds2)),'m+-');
        hold on;
        ylim([-4,7]);xlim([-2,9]);
        plot(log(temps(inds)),log(meta24.o2(inds2)),'m.-');
        plot(log(temps(inds)),log(meta2loc.o1(inds2)),'+-','Color',[1 .5 0]);
        plot(log(temps(inds)),log(meta2loc.o2(inds2)),'.-','Color',[1 .5 0]);
        plot(log(temps(inds)),log(meta4loc.o1(inds2)),'b+-');
        plot(log(temps(inds)),log(meta4loc.o2(inds2)),'b.-');
        legend('Min A to Min B','Min B to Min A','Min A to Min C','Min C to Min A','Min B to Min C','Min C to Min B','Location','northeast');
        title('AD Metastable Boundaries for Three Minima');
        xlabel('log(Temperature)');ylabel('log(alpha)');
        hold off;
        figure(1);
    end
    
    if 0
        %plot(log(temps(inds)),meta24.o1(inds2),'m+-');
        %hold on;
        xlim([-2,3.5]);
        %plot(log(temps(inds)),meta24.o2(inds2),'m.-');
        %plot(log(temps(inds)),meta2loc.o1(inds2),'+-','Color',[1 .5 0]);
        %hold on;
        %xlim([-2,3.5]);
        %plot(log(temps(inds)),meta2loc.o2(inds2),'.-','Color',[1 .5 0]);
        plot(log(temps(inds)),meta4loc.o1(inds2),'b+-');
        hold on;
        xlim([-2,3.5]);
        plot(log(temps(inds)),meta4loc.o2(inds2),'b.-');
        legend('Min B to Min C','Min C to Min B','Location','northeast');%,'Min A to Min C','Min C to Min A','Min B to Min C','Min C to Min B','Location','northeast');
        title('AD Metastable Boundary: Min B and Min C');
        xlabel('log(Temperature)');ylabel('alpha');
        hold off;
        figure(1);
    end
    
     if 1
        plot(log(temps(inds)),meta24.barz1(inds2,1),'m+-');
        hold on;
        xlim([-2,6.5]);ylim((10^5)*[-7.23,-6.87]);
        plot(log(temps(inds)),meta24.barz2(inds2,1),'m.-');
        plot(log(temps(inds)),meta2loc.barz1(inds2,1),'+-','Color',[1 .5 0]);
        plot(log(temps(inds)),meta2loc.barz2(inds2,1),'.-','Color',[1 .5 0]);
        plot(log(temps(inds)),meta4loc.barz1(inds2,1),'b+-');
        plot(log(temps(inds)),meta4loc.barz2(inds2,1),'b.-');
        plot([-100,100],[get_energy(net,config,min2),get_energy(net,config,min2)],':','Color',[.4,.4,.4],'LineWidth',1.75);
        plot([-100,100],[get_energy(net,config,min4),get_energy(net,config,min4)],'-.','Color',[.4,.4,.4],'LineWidth',1.2);
        plot([-100,100],[get_energy(net,config,min_loc),get_energy(net,config,min_loc)],'--','Color',[.4,.4,.4],'LineWidth',1.2);
        legend('Min A to Min B','Min B to Min A','Min A to Min C','Min C to Min A','Min B to Min C','Min C to Min B','Energy of Min A','Energy of Min B','Energy of Min C','Location','northwest');
        title('AD Barrier Estimation for Three Minima');
        xlabel('log(Temperature)');ylabel('Estimated Barrier Energy');
        hold off;
        figure(1);
     end
    
     if 0
         [inter_ensAB,inter_imsAB] = get_interpolation_energys(net,config,min2,min4,150);
         [inter_ensAC,inter_imsAC] = get_interpolation_energys(net,config,min2,min_loc,150);
         [inter_ensBC,inter_imsBC] = get_interpolation_energys(net,config,min4,min_loc,150);       
         for i = 1:14
             imwrite(inter_imsAB(:,:,i*10)/256,['/Users/mitch/Documents/MCMC Material/ELM Paper/figures/inter_imAB',num2str(i),'.png']);
             imwrite(inter_imsAC(:,:,i*10)/256,['/Users/mitch/Documents/MCMC Material/ELM Paper/figures/inter_imAC',num2str(i),'.png']);
             imwrite(inter_imsBC(:,:,i*10)/256,['/Users/mitch/Documents/MCMC Material/ELM Paper/figures/inter_imBC',num2str(i),'.png']);
         end
         
     end
end