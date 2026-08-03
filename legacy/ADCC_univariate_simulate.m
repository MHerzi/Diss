function [simulatedata, H] = ADCC_univariate_simulate(parameters, data, p, o, q, garchtype, errortype, stdEstimate)
% PURPOSE:
%     Make a univariate time series of conditional variances for use by ADCC_FULL_LIKELIHOOD
% 
% 
% USAGE:
%    [simulatedata, H] = ADCC_univariate_simulate(parameters,p,q,data)
% 
% 
% INPUTS:
%     parameters    - A vector(p+q+1) by 1 of parameters of form 
%                        [omega archp(1) ... archp(p) garchp(1) ... garchp(q)]
%     p             - The number of innovations to include(scalar)
%     q             - The length of the AR in the Garch process(scalar)
%     data          - A set of zero mean residuals you wish to introduce garch effects to
% 
% OUTPUTS:
%      simulatedata - The data that has had garch standard deviation applied to it
%      H            - The conditional variances for the data
% 
% 
% COMMENTS:
% 
% 
% Author: Kevin Sheppard, Martin Grziska
% kevin.sheppard@economics.ox.ac.uk

if isempty(q)
   m=p;
else
   m  =  max(p,q);   
end

[t,k]=size(data);
UncondStd =  sqrt(cov(data));
h=UncondStd.^2*ones(t+m,1);
data=[UncondStd*ones(m,1);data];
RandomNums=randn(t+m,1);
T=size(data,1);

parameters(find(parameters(1:1+p+o+q) <= 0)) = realmin;
garchparameters=parameters(1:p+o+q+1);
remainingparams=parameters(p+q+o+2:length(parameters));
constp=garchparameters(1);
archp=garchparameters(2:p+1);
tarchp=garchparameters(p+2:p+o+1);
garchp=garchparameters(p+o+2:p+q+o+1);

if garchtype == 1
   lambda=2;
   nu=2;
   b=0;
elseif garchtype == 2
   lambda=1;
   nu=1;
   b=0;
elseif garchtype == 3
   lambda=1;
   nu=1;
   b=remainingparams(1);
elseif garchtype == 4
   lambda=remainingparams(1);
   nu=lambda;
   b=0;
elseif garchtype == 5
   lambda=2;
   nu=2;
   b=remainingparams(1);
elseif garchtype == 6
   lambda=remainingparams(1);
   nu=lambda;
   b=0;
elseif garchtype == 7
   lambda=remainingparams(1);
   nu=lambda;
   b=remainingparams(2);
elseif garchtype == 8
   lambda=2;
   nu=2;
   b=0;
end


if isempty(q)
   m=p;
else
   m  =  max(p,q);   
end
T           =  size(data,1);
t = (m + 1):T;
datamb=data-b;
dataneg=(data<0).*datamb;
h=multigarchcore(abs(datamb),abs(dataneg),garchparameters,nu,lambda,b,p,o,q,m,T,stdEstimate);
h=h.^2;
simulatedata=data((m+1):T);
H=h((m + 1):T);

