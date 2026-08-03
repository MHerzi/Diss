function [c, ceq] = DCC_nonlincon(params, stdresid, P, Q, epsilon);

% PURPOSE:
%        Nonlinear constraint function for use in the DCC_MVGARCH estimation 
%        Qinitial has to be positive-definite to ensure thate Rt is
%        positive definite
% 
% USAGE:
%        [c, ceq] = AGDCC_nonlincon(params, stdresid, P, Q, epsilon)
% 
% INPUTS:
%    params      - A (2*P+Q)*k by 1 vector of parameters of the form [dccPparameters; adccPparameters, dccQparameters]
%    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
%    epsilon     - 
% 
% OUTPUTS:
%    c              - The inequality constraint
%    ceq            - The equality constraint

% 
% COMMENTS:
% Stationarity for the AGDCC model is fulfilled if the matrix Qinitial is
% positive definite, see Urga (2007),  A multivariate model for the
% non-normal behavior of Financial Assets;
% http://www.cass.city.ac.uk/conferences/measuringdependence/Files/presenta
% tions/urga.pdf
% 
% 
% Author: Martin Grziska.
% last modification: 09/01/2010

a = diag(params(1:P));       
b = diag(params(P+1:P+Q));      

c = a^2 + b^2-(1-epsilon);
ceq = 0;
