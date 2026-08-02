function [dccparameters, dccllf, EXITFLAG] = Est_DCC(data,dccP,dccQ)
% PURPOSE:
%        Estimates a multivariate GARCH model using the DCC estimator of Engle and Sheppard
% 
% USAGE:
%        [dccparameters, dccllf, EXITFLAG] = Est_DCC(data,dccP,dccQ)
% 
% INPUTS:
%      data          = A zero mean t by k vector of residuals from some filtration
%      dccP          = The lag length of the innovation term in the DCC estimator
%      dccQ          = The lag length of the lagged correlation matrices in the DCC estimator
%      archP         = One of three things:    Empty   in which case a 1 innovation model is estimated for each series
%                                      A scalar, p     in which case a p innovation model is estimated for each series
%                                      A k by 1 vector in which case the ith series has innovation terms p=archP(i)
%      garchQ        = One of three things:    Empty   in which case a 1 GARCH lag is used in estimation for each series
%                                      A scalar, q     in which case a q GARCH lags is used in estimation for each series
%                                      A k by 1 vector in which case the ith series has lagged variance terms q=archQ(i)
%      epsilon       = scalar (small number; ensures that a+b<1)
% 
% OUTPUTS:
%      parameters    = A vector of parameters estimated form the model of the form
%                          [GarchParams(1) GarchParams(2) ... GarchParams(k) DCCParams]
%                          where the garch parameters from each estimation are of the form
%                          [omega(i) alpha(i1) alpha(i2) ... alpha(ip(i)) beta(i1) beta(i2) ... beta(iq(i))]
%      loglikelihood = The log likelihood evaluated at the optimum
%      Ht            = A k by k by t array of conditional variances
%      Qt            = A k by k by t array of Qt elements
%      likelihoods   = the estimated likelihoods t by 1
%      stderrors     = A length(parameters)^2 matrix of estimated correct standard errors
%      A             = The estimated A form the rebust standard errors
%      B             = The estimated B from the standard errors
%      scores        = The estimated scores of the likelihood t by length(parameters)
% 
% 
% COMMENTS:
% 
% 
% Author: MArtin Grziska based on a code of Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 2    Date: 09/01/2010


options  =  optimset('fmincon');
options  =  optimset(options , 'Display'     , 'iter');
options  =  optimset(options , 'Diagnostics' , 'off');
options  =  optimset(options , 'LevenbergMarquardt' , 'on');
options  =  optimset(options , 'LargeScale'  , 'off');
options  =  optimset(options , 'MaxIter',500,'MaxFunEvals',10000);
options  =  optimset(options , 'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6);


dccstarting=[ones(1,dccP)*.01/dccP ones(1,dccQ)*.97/dccQ]';
fprintf(1,'\n\nEstimating the DCC model\n')

epsilon= 1e-5;

lower = [zeros(1,dccP) zeros(1,dccQ)]+1e-6;
upper = [ones(1,dccP) ones(1,dccQ)]-1e-6;

count = 1;
EXITFLAG = 0;

while count<=10 && EXITFLAG<1
    epsilon = epsilon/2;
    [dccparameters,dccllf,EXITFLAG,OUTPUT,LAMBDA,GRAD]=fmincon('dcc_mvgarch_likelihood_grm',dccstarting,[],[],[],[],lower,upper,'DCC_nonlincon',options,data,dccP,dccQ,epsilon);
    count = count + 1;
    if dccllf==1e+009
        EXITFLAG = 0;
    end
end

if EXITFLAG<=0 || dccllf==1e+009
    options = optimset(options, 'Algorithm','interior-point');
    [dccparameters,dccllf,EXITFLAG,OUTPUT,LAMBDA,GRAD]=fmincon('dcc_mvgarch_likelihood_grm',dccstarting,[],[],[],[],lower,upper,'DCC_nonlincon',options,data,dccP,dccQ,epsilon);
end

