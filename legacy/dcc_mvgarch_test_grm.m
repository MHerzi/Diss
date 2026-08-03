function [pval, stat]=dcc_mvgarch_test_grm(stdresid,nlags)
% PURPOSE:
%     Test for presence of dynamic correlation
% 
% USAGE:
%     [pval, stat]=dcc_mvgarch_test(data,nlags);
% 
% INPUTS:
%     data  - T by k matrix of (modification:standardized)residuals to be tested or dynamic corrrelation
%     archP - The length of the news terms in each univariate garch(either a scalar or a k by 1 vector)
%     garchQ - The length of the smoothing terms in each univariate garch(either a scalar or a k by 1 vector)
%     nlags - THe number of lags to use in the test
% 
% OUTPUTS:
%     pval - The probability the correlation is constant
%     stat - The Chi^2 stat, with nlags+1 D.F>
% 
% COMMENTS:
% 
% 
% Author: Kevin Sheppard, modified by Martin Grziska
% kevin.sheppard@economics.ox.ac.uk
% Revision: 2    Date: 12/31/2001
% modification: 06/10/2010


[t,k]=size(stdresid);

outerprods=[];
for i=1:k
    for j=i+1:k;
        outerprods=[outerprods stdresid(:,i).*stdresid(:,j)];        
    end
end

j=size(outerprods,2);

regressors=[];
regressand=[];
for i=1:j
    [Y,X]=newlagmatrix(outerprods(:,i),nlags,1);
    regressors=[regressors; X];
    regressand=[regressand; Y];
end
beta=regressors\regressand;
XpX=(regressors'*regressors);
e=regressand-regressors*beta;
sig=e'*e/(length(regressors-nlags-1));
stat=beta'*XpX*beta/sqrt(sig);
pval=1-chi2cdf(stat,nlags+1);
