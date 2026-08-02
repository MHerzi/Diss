function [c,ceq] = Nonlincon_vine(params, data, CopulaSpec, epsilon)
% Helper-function für VineStartinvals.m; nicht-lineare constraints für
% DCC-Modelle
% Achtung: DCC-Matrizen dürfen jeweils nur 1 Lag haben!!
%
% Author: Martin Grziska
% Datum: 03.09.2010


if strcmp(CopulaSpec.type,'Gaussian') || strcmp(CopulaSpec.type,'Clayton') || strcmp(CopulaSpec.type,'Rotated Clayton')||  strcmp(CopulaSpec.type,'Gumbel')
    stdresid = norminv(data);
elseif strcmp(CopulaSpec.type,'t')
    stdresid = tinv(data,params(end));
end



% es kann eine feste Zahl für die Auswahl der Parameter vergeben werden, da
% der Datensatz bei vine-copulas immer bivariat ist

if strcmp(CopulaSpec.corrspec,'DCC')
    % Constraints nach Cappiello et al (2006): a^2+b^2<1
    a = params(1);
    b = params(2);    
    c = a^2 + b^2 - (1-epsilon);
    ceq=[];
elseif strcmp(CopulaSpec.corrspec,'ADCC')
    a = params(1);
    g = params(2);
    b = params(3);
    neg_stdresid = stdresid.*(stdresid<0);
    Qbar = cov(stdresid);
    Nbar = cov(neg_stdresid);
    Qinitial = Qbar* (1 - a^2 - b^2) - Nbar*g^2;
    delta = max(eig(Qbar^(-.5)*Qinitial*Qbar^(-.5)));
    nonlin= a^2 + b^2 + delta*g^2 ;    
    c = max(nonlin) - (1-epsilon);
    ceq = [];
elseif strcmp(CopulaSpec.corrspec,'GDCC')
    A = diag(params(1:2));
    B = diag(params(3:4));
    Qbar = cov(stdresid);
    Qinitial = Qbar - A'*Qbar*A - B'*Qbar*B;
    x_1 = eig(Qinitial);
    x_1 = x_1 / abs(max(x_1));
    c = min(x_1)*(-1) + epsilon;
    ceq = 0;
elseif strcmp(CopulaSpec.corrspec,'AGDCC')
    A = diag(params(1:2));
    G = diag(params(3:4));
    B = diag(params(5:6));
    neg_stdresid = stdresid.*(stdresid<0);
    Qbar = cov(stdresid);
    Nbar = cov(neg_stdresid);
    Qinitial = Qbar - A'*Qbar*A - B'*Qbar*B - G'*Nbar*G;
    x_1 = eig(Qinitial);
    x_1 = x_1 / abs(max(x_1));
    c = min(x_1)*(-1) + epsilon;
    ceq = 0;
end

% stelle sicher, dass es eine Lösung gibt
if ~isreal(c) || isnan(c) || isinf(c)
    c=0;
end


