function [c,ceq] = nonlincon_Mix(weight,data)

c = [];
ceq = weight(1)+weight(2)-1+1e-5;