function invu2 = inv_hfunc_gumbel(w,u1,u2,theta)

% Helfer-Funktion für SimVine.m.  Gibt die Gleichung vor, die gelöst wird
% um den numerischen Wert der inversen h-Funktion für die Gumbel-Copula zu
% erhalten. Ursprungsgleichung w = h(u1,u2,theta); Gleichung in der
% Funktion unten: h(u1,u2,theta) - w = 0. Diese Glecihung wird letzendlich
% nach u2 gelöst
% 
% USAGE:
%       invu2 = inv_hfunc_gumbel(w,u1,u2,theta)
% 
% INPUTS:
%        w: unif(0,1)-Variable
%       u1: unif(0,1)-Variable
%       u2: unif(0,1)-Variable
%    theta: Abhängigkeitsparameter der Gumbel-Copula
% 
% OUTPUTS:
%    invu2: unif(0,1)-Variable
% 
% Author: Martin Grziska,      August,08,2010

h1 = exp(-((-log(u1)).^theta + (-log(u2)).^theta).^(1/theta)).* 1./u2 .* (-log(u2)).^(theta-1);
h2 = ((-log(u1)).^theta+(-log(u2)).^theta).^(1/theta-1);
h = h1.*h2;
invu2 = h - w ; 
