% nicht linearen Constraint für archimedische dynamische Copula definieren
%
% INPUT:
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
% OUTPUT:
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function [c, ceq] = nonLinCon_ArchCop_tv_2(tv_faktor, CopParam_1, family, data, P, Q, Dynamic, Density_Func_in, epsilon);

family = lower(family);
nc = numel(str2double(family)); % Anzahl der Copulas auslesen

[t,k]=size(data);

% Beschränken der ARMA Parameter der archimedischen Copula auf ein
% vernünftiges Maß. Dabei soll vermieden werden dass völlig übertriebene
% Prognosen entstehen. Der archimedische CopulaParameter läuft bei
% horizon=inf gegen K/(1-AR) -> Die Exponentialtransformationen dieses
% Wertes muss irgendwo zwischen [0 3.5] einpendeln.
if nc > 1
    np = 0; % Filtern welche Parameter zum ARMA Prozess der archimedischen Copula gehören
    for j = 1:nc
        switch family{j}
            case 't'
                if strcmp(Dynamic,'ADCC')
                    np = np+4;
                elseif strcmp(Dynamic,'DCC')
                    np = np+3;
                elseif strcmp(Dynamic,'GDCC')
                    np = np+P*k+Q*k+1;
                elseif strcmp(Dynamic,'AGDCC')
                    np = np+2*P*k+Q*k+1;
                end
            case 'gaussian'
                if strcmp(Dynamic,'ADCC')
                    np = np+3;
                elseif strcmp(Dynamic,'DCC')
                    np = np+2;
                elseif strcmp(Dynamic,'GDCC')
                    np = np+P*k+Q*k;
                elseif strcmp(Dynamic,'AGDCC')
                    np = np+2*P*k+Q*k;
                end
            case {'clayton', 'gumbel'}
                break
        end
    end
    ACDP = tv_faktor(np+1:np+3); %archimedische dynamische CopulaParameter
else
    ACDP = tv_faktor;
end

% Sicherstellen dass der ARMA Prozess so kalibriert wird
% dass eine vernünftige Prognose des Copulaparameters
% gewährleistet ist. Da die Prognose des ARMA Modells
% exponentiert wird, muss K/(1-AR) < 3.5 eingehalten
% werden. Dies wird anhand der Ungleichungen gemacht.
K = ACDP(1);
AR = ACDP(2);
MA = ACDP(3);

% % original-constraint
% c = K./(1-AR) - 3.5;

% % abs(AR) und abs(MA) m�ssen jeweils <1 sein
c(1,1) = abs(AR) - (1-1e-6);
c(2,1) = abs(MA) - (1-1e-6);

ceq = 0;

