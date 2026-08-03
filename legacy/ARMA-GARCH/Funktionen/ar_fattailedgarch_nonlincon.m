function [c,ceq] =  ar_fattailedgarch_nonlincon(params,data,p,q,errortype,arlag,const)
% Helper function for EGARCH-SKEWT estimation:
% 1. constraint:  sum(abs(AR-coefficients)) <1


c(:,1) = sum(abs(params(1+p+q+1:1+p+q+arlag)))-1 + 1e-5;

ceq = 0;

% Author: Martin Grziska 03/02/2010