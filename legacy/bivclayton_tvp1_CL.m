function [CL,kappa] = bivclayton_tvp1_CL(theta,data,kappabar)
% function CL = claytonCL(theta,data)
%
% The negative copula log-likelihood of a
% member of Clayton's family
% From Joe(1997), p141: Family B4 Kimeldorf and Sampson
%
% INPUTS:
%               theta: vector of starting values
%				data:  tx2 data matrix of unif variables
%
% 07.02.2008
% last modification: 03/11/2010
%
% Author: Martin Grziska



T = size(data,1);
u = data(:,1);
v = data(:,2);

kappa = -999.99*ones(T,1);
kappa(1) = kappabar;
for jj = 2:T;
    if jj<=10
        psi = theta(1)+ theta(2)*kappa(jj-1) + theta(3)*(mean(abs(u(1:jj-1)-v(1:jj-1))));
    else
        psi = theta(1)+ theta(2)*kappa(jj-1) + theta(3)*(mean(abs(u(jj-10:jj-1)-v(jj-10:jj-1))));
    end
    kappa(jj) = (1+exp(-psi))^(-1);
    kappa(jj) = copulaparam('Clayton',kappa(jj));
end

CL = log(1+kappa) - (kappa+1).*(log(u)+log(v));
CL = CL - (2+1./kappa).*log((u.^(-kappa)) + (v.^(-kappa)) -1);
CL = sum(CL);
CL = -CL;


if isreal(theta)==0;
   CL = 1e6;
elseif isreal(CL)==0;
   CL = 1e7;
elseif isnan(CL)
   CL = 1e8;
elseif isinf(CL)
   CL = 1e9;
end
