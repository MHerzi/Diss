function [c,ceq] =  ar_egarch_skewt_nonlincon(params,data,p,q,o,stdEstimate,errortype,arlag)
% Helper function for skew-t estimation:
% 1. constraint:  -1<lambda<1
% 2. constraint:  sum(abs(AR-coefficients)) <1

c(:,1) = 2*abs(params(end))-2 + 1e-5;
c(:,2) = sum(abs(params(1+p+o+q+1:1+p+o+q+arlag))) - 1 + 1e-5;
 
ceq = 0;

% Author: Martin Grziska 03/02/2010