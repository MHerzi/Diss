function [LLF, h, likelihoods, resid] = ar_avgarch_garchlikelihood_grm(parameters , data , p , q, errortype, arlag, const)
% PURPOSE:
%     Likelihood for fattailed garch estimation
%
% USAGE:
%     [LLF, h, likelihoods] = fattailed_garchlikelihood(parameters , data , p , q, errortype, stdEstimate, T)
%
% INPUTS:
%     parameters:   A vector of GARCH process aprams of the form [constant, arch, garch]
%     data:         A set of zero mean residuals
%     p:            The lag order length for ARCH
%     q:            The lag order length for GARCH
%     m:            The max of p and q
%     error:        The type of error being assumed, valid types are:
%                   1 if 'NORMAL'
%                   2 if 'STUDENTST'
%                   3 if 'GED'
%     stdEstimate:  The std deviation of the data
%     T:             Length of data
%     arlag: laglength of AR(k)-model
%     const : 1 - estimate AR(k) with const, else 0
%
% OUTPUTS:
%     LLF:          Minus 1 times the log likelihood
%     h:            Time series of conditional volatilities
%     likelihoods   Time series of likelihoods
%
% COMMENTS:
%     This is a helper function for garchinmean
%
%
% Author: Kevin Sheppard, Martin Grziska
% kevin.sheppard@economics.ox.ac.uk
% Revision: 2    Date: 12/31/2001
% Modification: 02/21/2010
%

% AR(k)-model
% form of parameters: [omega, alpha, talpha; beta; AR; (nu)];

if errortype == 1
    b0 = parameters(1+p+q+1:end);
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
    parameters = parameters(1:1+p+q);
else
    b0 = parameters(1+p+q+1:end-1);
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
    parameters = [parameters(1:1+p+q); parameters(end)];
end

stdEstimate = (std(resid))^2;
resid=[stdEstimate;resid];

T=length(resid);


[r,c]=size(parameters);
if c>r
    parameters=parameters';
end


parameters(find(parameters <= 0)) = realmin;

constp=parameters(1);
archp=parameters(2:p+1);
garchp=parameters(p+2:p+q+1);
if errortype ~=1;
    nu = parameters(p+q+2);
    parameters = parameters(1:p+q+1);
end



if isempty(q)
    m=p;
else
    m  =  max(p,q);
end


h=avgarchcore(resid,parameters,stdEstimate,p,q,m,T);
h = h.^2;
Tau = T-m;
LLF = 0;
t = (m + 1):T;
if errortype == 1
    LLF  =  sum(log(h(t))) + sum((resid(t).^2)./h(t));
    LLF  =  0.5 * (LLF  +  (T - m)*log(2*pi));
elseif errortype == 2
    LLF = Tau*gammaln(0.5*(nu+1)) - Tau*gammaln(nu/2) - Tau/2*log(pi*(nu-2));
    LLF = LLF - 0.5*sum(log(h(t))) - ((nu+1)/2)*sum(log(1 + (resid(t).^2)./(h(t)*(nu-2)) ));
    LLF = -LLF;
else
    Beta = (2^(-2/nu) * gamma(1/nu)/gamma(3/nu))^(0.5);
    LLF = (Tau * log(nu)) - (Tau*log(Beta)) - (Tau*gammaln(1/nu)) - Tau*(1+1/nu)*log(2);
    LLF = LLF - 0.5 * sum(log(h(t))) - 0.5 * sum((abs(resid(t)./(sqrt(h(t))*Beta))).^nu);
    LLF = -LLF;
end


if nargout >1
    if nargout > 2
        likelihoods=zeros(size(T));
        if errortype == 1
            likelihoods = 0.5 * ((log(h(t))) + ((resid(t).^2)./h(t)) + log(2*pi));
            likelihoods = -likelihoods;
        elseif errortype == 2
            likelihoods = gammaln(0.5*(nu+1)) - gammaln(nu/2) - 1/2*log(pi*(nu-2))...
                - 0.5*(log(h(t))) - ((nu+1)/2)*(log(1 + (resid(t).^2)./(h(t)*(nu-2)) ));
            likelihoods = -likelihoods;
        else
            Beta = (2^(-2/nu) * gamma(1/nu)/gamma(3/nu))^(0.5);
            likelihoods = (log(nu)/(Beta*(2^(1+1/nu))*gamma(1/nu))) - 0.5 * (log(h(t))) ...
                - 0.5 * ((abs(resid(t)./(sqrt(h(t))*Beta))).^nu);
            likelihoods = -likelihoods;
        end
    end
    h=h(t);
end

if isreal(parameters)==0;
   LLF = 1e6;
elseif isreal(LLF)==0;
   LLF = 1e7;
elseif isnan(LLF)
   LLF = 1e8;
elseif isinf(LLF)
   LLF = 1e9;
end


resid = resid(t);




