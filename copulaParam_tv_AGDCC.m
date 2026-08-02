% Berechnen der zeitvariablen Korrelationsmatrizen nach Capiello et al.
% (2006). Skalarversion.
%
% INPUT:
% - tv_faktor: Parameter des ADCC Prozesses der Korrelationsmatrizen
% - data_inv: [0 1] invertierte Zeitreihen
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% - funktion: entweder 'kalibrieren' oder 'vorhersage'
% - horizon: Zeithorizont der Vorhersage
%
% OUTPUT:
% - Rt: zeitvariable Korrelationsmatrix
%
%   Author: Valentin Braun, Martin Grizska
%   Phd Student in finance Goethe Universität, LMU
function [R_tv] = copulaParam_tv_AGDCC(tv_faktor, P, Q, data_inv, funktion, horizon);

funktion = lower(funktion);
tv_faktor = tv_faktor(:)';
[s1 s2] = size(data_inv);

% Vorhersage Horizont wird für MLE auf 0 gesetzt
if strcmp(funktion, 'kalibrieren')
	horizon = 0; 
end

% Faktoren des DCC Modells übergeben
A = diag(tv_faktor(1:P*s2)); % ARCH Faktor
G = diag(tv_faktor(P*s2+1:2*P*s2));
B = diag(tv_faktor(2*P*s2+1:2*P*s2+Q*s2)); % GARCH Faktor

% First compute Qbar, the correlation matrix of the standardized residuals,
% and Nbar, the correlation matrix of the standardized residuals,
% conditional on the standardized residuals being negative
Qbar = cov(data_inv); % statische Kovarianzmatrix der stand. Residuen
Nbar = cov(data_inv.*(data_inv < 0));
Qinitial = Qbar - A'*Qbar*A - B'*Qbar*B - G'*Nbar*G;
m = max(P,Q);

Qt = zeros(s2,s2,s1+m+horizon);
R_tv = zeros(s2,s2,s1+m+horizon);
Qt(:,:,1:m+horizon) = repmat(Qbar,[1 1 m+horizon]);
R_tv(:,:,1:m+horizon) = repmat(Qbar,[1 1 m+horizon]);
violatePSD = 0; % Kontrollvariable für positive Semidefinitheit
% Erwartungswert der Residuen ist 0. Daher werden für horizon-1
% Zeiteinheiten 0 an die gefilterten Residuen angehängt
stdresid = [zeros(m,s2); data_inv; zeros(horizon, s2)];
negativeStdresid =[zeros(m,s2); data_inv .* (data_inv < 0); zeros(horizon, s2)];

for j = (m+1):s1+m+horizon
    Qt(:,:,j) = Qinitial;
    for i=1:P
        Qt(:,:,j) = Qt(:,:,j) + A'*(stdresid(j-i,:)'*stdresid(j-i,:))*A; % ARCH Faktor
        Qt(:,:,j) = Qt(:,:,j) + G'*(negativeStdresid(j-i,:)'*negativeStdresid(j-i,:))*G;
    end
    for i = 1:Q
        Qt(:,:,j) = Qt(:,:,j) + B'*Qt(:,:,j-i)*B; % GARCH Faktor
    end
    R_tv_emp = Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
    R_tv_emp = R_tv_emp - diag(diag(R_tv_emp)) + eye(s2); % Sicherstellen dass die Diagonale nur exate 1 beinhält, sonst wäre dies eine Korrelationsmatrix
    R_tv(:,:,j) = R_tv_emp;
    maxmax = max(max(R_tv(:,:,j)));
    minmin = min(min(R_tv(:,:,j)));
    if maxmax > 1 || minmin < -1
        violatedPSD = 1;
    end
end

% Output vorbereiten
if strcmp(funktion, 'kalibrieren')
    R_tv = R_tv(:,:,(m+1:s1+m));
elseif strcmp(funktion, 'vorhersage')
    R_tv = R_tv(:,:,(s1+m+1:s1+m+horizon));
end
            
            