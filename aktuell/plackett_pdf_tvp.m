function [pdf,rhohat] = plackett_pdf_tvp(u,v,theta,rhobar)
% function pdf = plackett_pdf(u,v,k)
%
% Computes the value of the Plackett copula pdf at a specified point
% 
% INPUTS:	U, a Tx1 vector (or a scalar) of F(X[t])
%				V, a Tx1 vector (or a scalat) of G(Y[t])
%				K, a Tx1 vector (or a scalar) of kappas
%
% 30.01.2009
% Martin Grziska


% Written for the following papers:
%
% Patton, A.J., 2006, Modelling Asymmetric Exchange Rate Dependence, International Economic Review, 47(2), 527-556. 
% Patton, A.J., 2006, Estimation of Multivariate Models for Time Series of Possibly Different Lengths, Journal of Applied Econometrics, 21(2), 147-173.  
% Patton, A.J., 2004, On the Out-of-Sample Importance of Skewness and Asymmetric Dependence for Asset Allocation, Journal of Financial Econometrics, 2(1), 130-168. 
%
% http://fmg.lse.ac.uk/~patton


T = max([size(u,1),size(v,1),size(theta,1)]);

% stretching the input vectors to match
if size(u,1)<T;
   u = u*ones(T,1);
end
if size(v,1)<T;
   v = v*ones(T,1);
end

kappa = -999.99*ones(T,1);
kappa(1) = rhobar;			% this is the MLE of kappa in the time-invariant version of this model
for jj = 2:T
    if jj<=10
        kappa(jj) = theta(1) + theta(2)*mean(u(1:jj-1).*v(1:jj-1)) + theta(3)*kappa(jj-1);
    else
        kappa(jj) = theta(1) + theta(2)*mean(u(jj-10:jj-1).*v(jj-10:jj-1)) + theta(3)*kappa(jj-1);
    end
end
rhohat = kappa;  % time-path of conditional copula parameter

pdf = kappa.*(1+(u - 2*u.*v + v).*(kappa-1))./((((1+(kappa-1).*(u+v)).^2) - 4*u.*v.*kappa.*(kappa-1)).^(3/2));