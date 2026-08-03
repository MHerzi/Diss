% Non-Lineare Constraints für dynamische elliptische Copulas mit
% Skalarparametern; Capiello et al. (2006), Skalarversion
%
% INPUT:
% - tv_faktor: Parameter der dynamischen lliptischen Copula
% - CopParam_1: bei archimedischen Copulas: erster CopulaParameter im Zeitablauf
%                            bei elliptischen Copulas bleibt dieser leer: []
% - weights_1: Startwerte der Gewichte
% - family: t, gaussian
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% - data: [0 1] verteilte Zeitreihen
%
%   Author: Valentin Braun, Martin Grziska
%   Phd Student in finance Goethe Universität, LMU München
function [c ceq] = nonLinCon_Skalar_tv(tv_faktor, CopParam_1, family, data, P, Q);

family = lower(family);
epsilon = 1-10^(-3);
[s1 s2] = size(data);
stdresid = zeros(s1,s2);
        
switch family
    case 't'
        DoF = tv_faktor(end);
        for i = 1:s2
            stdresid(:,i) = tinv(data(:,i),DoF);
        end
    case 'gaussian'
        for i = 1:s2
            stdresid(:,i) = norminv(data(:,i),0,1);
        end
end

a = tv_faktor(1:P);       % ARCH for all residuals
g = tv_faktor(P+1:2*P);   % ARCH for negative residuals
b = tv_faktor(2*P+1:2*P+Q);        % GARCH
sumA = sum(a);
sumA_negative = sum(g);
sumB = sum(b);

% Compute Qbar, the correlation matrix of the standardized residuals,
% and Nbar, the correlation matrix of the standardized residuals,
% conditional on the standardized residuals being negative
Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

% Formel nach Cappiello et al (2006) "Asymmetric Dynamics in Correlations
% of Global Equity and Bond Returns", S.543 f.
x_1 = eig(Qbar^(-1/2)*Nbar_negative*Qbar^(-1/2));
x_1 = max(x_1);

c = (sumA + sumB + x_1*sumA_negative) - (1+epsilon);
ceq = 0;






















