% Non-Lineare Constraints für dynamische Mixture-Copulas mit
% Skalarparametern; Capiello et al. (2006), Skalarversion
%
% INPUT:
% - tv_faktor: Parameter der dynamischen lliptischen Copula
% - CopParam_1: bei archimedischen Copulas: erster CopulaParameter im Zeitablauf
%                            bei elliptischen Copulas bleibt dieser leer: []
% - family: t, gaussian
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% - data: [0 1] verteilte Zeitreihen
% - Density_Func_in: optionaler Input Parameter in dem die Dichtefunktion der
% Copula übergeben werden kann
%
% Hinweis:
% Die nicht linearen Constraints beziehen sich nur auf die elliptischen und
% Mixture Copulas. Dabei ist die Mixture Struktur darauf beschränkt nur 1
% elliptische Copula in der Kombination zuzulassen
% Für die archimedischen Copulas ist diese Funktion nur im Falle einer
% Mixture Struktur relevant. Dann wird hier kontrolliert dass die Parameter
% des ARMA Prozess in einem vernünftigen Rahmen bleiben
%
%   Author: Valentin Braun, Martin Grziska
%   Phd Student in finance Goethe Universität, LMU München
function [c ceq] = nonLinCon_Skalar_mix_tv_2(tv_faktor, CopParam_1, family, data, P, Q, Density_Func_in);

family = lower(family);
nc = numel(str2double(family)); % Anzahl der Copulas auslesen
epsilon = 1-10^(-3);
[s1 s2] = size(data);
stdresid = zeros(s1,s2);

% Auslesen der Position der elliptischen Copula (p_elliptic) in der Mixture Struktur
for p_elliptic = 1:nc
    if any(strcmp(family{p_elliptic}, {'t' 'gaussian'})) 
        break
    end
end
% Da archimedische Copulas immer 3 ZeitFaktoren nach Patton benötigen und
% nur eine ellitpische Copula pro Mixture zulässig ist, gibt nf an welcher
% ZeitFaktor als letzter den archimedischen Copulas zugeordnet werden muss.
% Die danach folgenden ZeitFaktoren gehören zu der verwendeten elliptischen
% Copula
% !!!!!!!!!!!!ACHTUNG!!!!!!!!!!!!
% die Variable p_elliptic darf NICHT überschrieben werden!
nf = (p_elliptic-1)*3; 

switch family{p_elliptic}
    case 't'
        DoF = tv_faktor(nf+2+P+Q); % DoF der t-Copula auslesen
        for i = 1:s2
            stdresid(:,i) = tinv(data(:,i),DoF);
        end
    case 'gaussian'
        for i = 1:s2
            stdresid(:,i) = norminv(data(:,i),0,1);
        end
end

% Platzhalter anlegen
[s1,s2] = size(stdresid);
a = tv_faktor(nf+1:nf+P); % ARCH for all residuals
g = tv_faktor(nf+P+1:nf+2*P); % ARCH for negative residuals -> entspricht dem Leverage Faktor des EGARCH Modells
b = tv_faktor(nf+2*P+1:nf+2*P+Q); % GARCH
% sumA = eye(s2)*sum(a);
% sumA_negative = eye(s2)*sum(g);
% sumB = eye(s2)*sum(b);

% Compute Qbar, the correlation matrix of the standardized residuals,
% and Nbar, the correlation matrix of the standardized residuals,
% conditional on the standardized residuals being negative
Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

% Next compute Qinitial
% Qinitial = Qbar * (1 - a - b) - Nbar_negative * g;

% Formel nach Cappiello et al (2006) "Asymmetric Dynamics in Correlations
% of Global Equity and Bond Returns", S.543 f.
x_11 = eig(Qbar^(-1/2)*Nbar_negative*Qbar^(-1/2));
x_1 = max(x_11);
c_cap = (a + b + x_1*abs(g)) -epsilon; % a,b,g sind beschränkt auf positive Werte
if ~isreal(c_cap)
    c_cap = 1; % Sicherstellen dass nur relle Zahlen übergeben werden
end


% Sicherstellen dass Koeffizienten der Korrelationsmatrizen im Bereich [-1 1] liegen
R_tv = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, stdresid, 'kalibrieren');
max_R_tv = max(reshape(R_tv,[], 1));
c_R_tv = max_R_tv - 1;
if ~isreal(c_R_tv)
    c_R_tv = 1; % Sicherstellen dass nur relle Zahlen übergeben werden
end

% Nonlinear Constraints für Gewichte Prozess kontrollieren: Konstante des Gewichte
% Prozess (k)
% k<= |ar|+|ma|
if nc > 1
    ARMA_param_W = tv_faktor(end-nc-2 : end-nc);
    k_W = ARMA_param_W(1); % Konstante des dynamischen Gewichtsprozess
    ar_W = ARMA_param_W(2); % AR-Faktor des dynamischen Gewichtsprozess
    ma_W = ARMA_param_W(3); % MA-Faktor des dynamischen Gewichtsprozess
    c_W = abs(ar_W) + abs(ma_W) - k_W;
else
    c_W = [];
    c_AC = [];
end

% Beschränken der ARMA Parameter der archimedischen Copulas in der Mixture Struktur auf ein
% vernünftiges Maß. Dabei soll vermieden werden dass völlig übertriebene
% Prognosen entstehen. Der archimedische CopulaParameter läuft bei
% horizon=inf gegen K/(1-AR) -> Die Exponentialtransformationen dieses
% Wertes muss irgendwo zwischen [0 3.5] einpendeln. 
if nc > 1
    c_AC = nonLinCon_ArchCop_tv_2(tv_faktor, CopParam_1, family, data, P, Q, Density_Func_in);
end
    
% Ausgabe der nonLin Constraint Resultate
c = [c_cap; c_R_tv; c_W; c_AC];
ceq = 0;



