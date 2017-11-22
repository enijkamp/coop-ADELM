function energy = get_im_energy(config,des_net,im)
    if ~isa(im,'single'), im = single(im); end
    im = im - des_net.normalization.averageImage;
    res = vl_simplenn(des_net,im);
    energy = sum(reshape(-(res(end).x),1,[]))+ 0.5*norm(im(:))^2/config.refsig1^2;
    clear res;
end