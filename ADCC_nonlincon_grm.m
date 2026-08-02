function [c, ceq] = ADCC_nonlincon_grm(params, stdresid, P, Q, epsilon)

% PURPOSE:
%        Nonlinear constraint function for use in the ADCC_MVGARCH estimation
%        Build as in Cappiello et al (see below)

% 
% USAGE:
%        [c, ceq] = ADCC_nonlincon_grm(params, stdresid, P, Q)
% 
% INPUTS:
%    params      - A 2*P+Q by 1 vector of parameters of the form [dccPparameters; dccQparameters, adccPparameters]
%    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
% 
% OUTPUTS:
%    c              - The inequality constraint
%    ceq            - The equality constraint

% 
% COMMENTS:
% 
% 
% Author: Martin Grziska

a          = params(1:P);       
a_negative = params(P+1:2*P);  
b = params(2*P+1:2*P+Q);        

% Compute Qbar, the correlation matrix of the standardized residuals,
% and Nbar, the correlation matrix of the standardized residuals,
% conditional on the standardized residuals being negative
Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

% Formel nach Cappiello et al (2006) "Asymmetric Dynamics in Correlations
% of Global Equity and Bond Returns", S.543 f.
x_1 = eig(Qbar^(-1/2)*Nbar_negative*Qbar^(-1/2));
x_1 = max(x_1);

c = (a^2 + b^2 + x_1*a_negative^2) - (1-epsilon);
ceq = 0;
