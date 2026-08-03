function [LLF, h, likelihoods, resid] = ar_skewt_avgarchlikelihood_grm(parameters , data , p , q, m, arlag, const)
% PURPOSE:
%     Likelihood for skewt_garch
%
% USAGE:
%     [LLF, grad, hessian, h, scores, robustse] = garchlikelihood(parameters , data , p , q, m, stdEstimate)
%
% INPUTS:
%     parameters:   A vector of GARCH process aprams of the form [constant, arch, garch]
%     data:         A set of zero mean residuals
%     p:            The lag order length for ARCH
%     q:            The lag order length for GARCH
%     m:            The max of p and q
%     stdEstimate:  The sample standard deviation of the data
%
% OUTPUTS:
%     LLF:          Minus 1 times the log likelihood
%     h:            The time series of conditional variances implied by the parameters and the data
%     likelihoods:  The time series of log likelioods
%
% COMMENTS:
%     This is a helper function for skewt_garch
%
% Author: Andrew Patton
% Author: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 2    Date: 12/31/2001
% Modification by Martin Grziska, 02/22/2010

b0 = parameters(1+p+q+1:1+p+q+1+arlag-1);
n = length(data);
if const == 1
    nlag = arlag-1;
    x = [ones(n,1) mlag(data,nlag)];
else
    nlag = arlag;
    x = mlag(data,nlag);
end
x = trimr(x,nlag,0);
data = trimr(data,nlag,0);
resid = data - x*b0;

stdEstimate = std(resid);
resid = [stdEstimate;resid];

[r,c]=size(parameters);
if c>r
    parameters=parameters';
end

% wenn ein paramtere keine reelle Zahl ist setzte Likelihood auf default
% Wert
if isnan(any(parameters))
    LLF = 1e-15;
else
    parameters(find(parameters(1:1+p+q) <= 0)) = realmin;
end


constp=parameters(1);
archp=parameters(2:p+1);
garchp=parameters(p+2:p+q+1);
nu = parameters(end-1);
lambda = parameters(end);

T           =  size(resid,1);
h=avgarchcore(resid,parameters(1:1+p+q),stdEstimate,p,q,m,T);
h = h.^2;

t=(m+1:T);

stdresid = resid(t)./sqrt(h(t));
LLF = 0.5*sum(log(h(t))) + ar_skewtdis_LL_grm([nu;lambda], stdresid);

% do some checks on the parameters and the LLF
if isnan(LLF)
    LLF=-1e15;
end
if isinf(LLF);
    LLF=1e-15;
end
if ~isreal(parameters)
    LLF=1e-15;
end
if isnan(parameters)
    LLF=1e-15;
end

if nargout>2
    [temp,likelihoods]=ar_skewtdis_LL_grm([nu;lambda], stdresid);
    likelihoods= likelihoods+0.5*log(h(t));
end

h=h(t);
resid=resid(t);

