function [c,ceq] =  ar_egarch_nonlincon(params,data,p,q,o,errortype,arlag,const)
%  ARMA(AR,MA)-EGARCH(P,O,Q) nonlinear parameter restriction.  Used in estimation of
%  EGARCH models.
%
%  USAGE:
%    [c,ceq] = ar_egarch_nonlincon(params,data,p,q,o,errortype,arlag,const)
%
%  INPUTS:
%    See ar_egarchEstLikelihood
%
%  OUTPUTS:
%    C    - Vector of nonlinear inequality constraints.  Based on the roots
%           of a polynomial in beta
%    CEQ  - Empty matrix
%
% Author: Martin Grziska 05/29/2010

% Übernommen aus MATLAB garchfit:
% The non-linear inequality constraints are those associated with
% the R roots of the auto-regressive (AR) polynomial and the M roots of
% the moving average (MA) polynomial. The polynomials are formed such
% that the roots computed are the eigenvalues. For an ARMA(R,M) model
% to be stationary and invertible, all eigenvalues must lie inside the
% unit circle of the complex plane.
% Sicherstellen dass die Summe der absoluten AR Parameter <1
% ist

if arlag>0
    AReigenValues =  roots([1 ; -params(1+p+o+q+1:1+p+o+q+arlag)]);
    AR_absSum = sum(abs(params(1+p+o+q+1:1+p+o+q+arlag))); % Summe der absoluten AR Parameter
else
    AReigenValues = [];
    AR_absSum = [];
end

% Restriktion des GARCH Parameters
GARCH_Param = params(1:1+p+o+q);
GARCHeigenValues = roots([1;-GARCH_Param]);

c_EIG = (abs([AReigenValues ; GARCHeigenValues]).^2) - .99998;
c_ABS = AR_absSum - .99998;
if errortype == 4
    c_SKEWT= abs(params(end)) - 0.99;
    c = [c_EIG; c_ABS; c_SKEWT];
else
    c = [c_EIG; c_ABS];
end
ceq = [];
