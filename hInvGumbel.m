function [hInv] = hInvGumbel(u1,u2,theta)

h = gumbel_cdf(u1,u2,theta)*1/u2*(-log(u2))^theta*((-log(u1))^theta+(-log(u2))^theta)^(1/theta-1);
hInv = 