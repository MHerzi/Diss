function [Tstat,robustSE] = TstatCopula(parameters, CopParam_1, family, data, P, Q)
% Computes robustSE and Tstat for different copula models
%
% USAGE: [Tstat,robustSE] = TstatCopula(parameters,data,P,Q,type)
%
% INPUTS:
%          parameters: vector containing parameters of copula
%          data:       t x k matrix of unif(0,1) variables
%          P:          lag length of ARCH and Leverage-Term
%          Q:          lag length of GARCH term
%          CopParam_1: unconditional CopParam (needed only for estimation)
%          family:     string can be either t, Gauss, Clayton, Gumbel 
%
% OUTPUTS:
%          robustSE: White(1982) robust standard errors
%          Tstat   : robust t-statistic for estimated parameters
%
% Author: Martin Grziska, 03/18/2010

[t,k] = size(data);
hess = hessian_2sided('copulaLL_tv',parameters, CopParam_1, family, data, P, Q);
stderrors=hess^(-1);
h = min(abs(parameters/2) + 1e-4,max(parameters,1e-2))*eps^(1/3);
hplus = parameters+h;
hminus = parameters-h;
likelihoodsplus = zeros(t,length(parameters));
likelihoodsminus = zeros(t,length(parameters));
for i=1:length(parameters)
    hparameters = parameters;
    hparameters(i) = hplus(i);
    [HOLDER, HOLDER1, indivlike] =  copulaLL_tv(hparameters, CopParam_1, family, data, P, Q);
    likelihoodsplus(:,i) = indivlike;
end
for i=1:length(parameters)
    hparameters = parameters;
    hparameters(i) = hminus(i);
    [HOLDER, HOLDER1, indivlike] =  copulaLL_tv(hparameters, CopParam_1, family, data, P, Q);
    likelihoodsminus(:,i) = indivlike;
end

scores = (likelihoodsplus-likelihoodsminus)./(2*repmat(h',t,1));
B=scores'*scores;
robustSE=stderrors*B*stderrors;
Tstat = parameters./diag(sqrt(robustSE));

