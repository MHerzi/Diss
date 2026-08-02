function h = hfunc_gumbel(data,theta)

u1 = data(:,1);
u2 = data(:,2);
h1 = gumbel_cdf(u1,u2,theta) .* 1./u2 .* (-log(u2)).^(theta-1);
h2 = ((-log(u1)).^theta+(-log(u2)).^theta).^(1/theta-1);
h = h1.*h2;