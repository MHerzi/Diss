function [c,ceq]=ar_multigarch_skewt_nonlincon(params, data, p , o, q, garchtype, errortype, stdEstimate, arlag)
% Helper function for ar_multigarch_grm: 
% 1. constraint -1<lambda<1
% 2. constraint: sum(abs(AR-coefficients)) <1

c(:,1) = 2*abs(params(end))-2 + 1e-5;
c(:,2) = sum(abs(params(1+p+q+o+1:1+p+q+o+arlag)))-1+1e-6;

ceq=0;