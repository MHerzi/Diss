function [c, ceq] = GDCC_nonlincon(params, stdresid, P, Q, epsilon)

% PURPOSE:
%        Nonlinear constraint function for use in the AGDCC_MVGARCH estimation 
%        Qinitial has to be positive-definite to ensure thate Rt is
%        positive definite
% 
% USAGE:
%        [c, ceq] = ADCC_nonlincon(params, stdresid, P, Q)
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
% Author: Martin Grziska, based on code of Christoph Schleicher
% last modification: August/27/2010

[t,k] = size(stdresid);
a = diag(params(1:(P*k)));       % ARCH for all residuals
b = diag(params((P*k)+1:(P*k)+Q*k));        % GARCH

% Qbar muss positive-definite sein
Qbar = cov(stdresid);
x1 = min(eig(Qbar));
if x1 <= 0
    error('Qbar ist nicht positive-definite')
end

% Damit Qt positive-definit ist, muss Qinitial positiv-semidefinit sein und so
% die Eigenwerte >=0; Fmincon setzt c<= 0; Laut Cappiello(2006) ECB-paper
% ist Qt mit Wahrscheinlichkeit 1 positiv definit, wenn Qinitial positiv
% semidefinit ist. 
Qinitial = Qbar - a'*Qbar*a - b'*Qbar*b;
x_1 = eig(Qinitial);
x_1 = x_1 / abs(max(x_1));
c = min(x_1)*(-1) + epsilon;
ceq = 0;

% alternative Spezifikation1: 
% % Engle(2002): jede matrix für sich muss positiv-definit sein, dann ist
% % auch Qt positiv-definit (gleiches gitl füt poisitiv semidefinit)
% % c(1) = min(eig(ones(k,k) - a*a' - b*b'))*(-1); 
% % c(2) = min(eig(a*a'))*(-1);
% % c(3) = min(eig(b*b'))*(-1);
% % ceq=[];

