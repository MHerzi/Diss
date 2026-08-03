% LL Function der dynamischen Mixture Copulas.
% Zulässige Copula Familien: t, Gaussian, Clayton, Gumbel
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - CopParam_1: bei archimedischen Copulas: erster CopulaParameter im Zeitablauf
%                            bei elliptischen Copulas wird dieser auf 0
%                            gesetzt
% - tv_faktor: Parameter des ARMA Prozesses des Copulaparameters und
% Startgewichte
% - data: [0 1] verteilte Zeitreihen
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% P,Q werden nur für die elliptischen Copulas übergeben!
% - Density_Func_in: optionaler Input Parameter in dem die Dichtefunktion der
% Copula übergeben werden kann
%
% OUTPUT:
% - LL: LogLikelihood Wert der einzelnen zeitvariablen Copulas
% - LL_Mix: LogLikelihood Wert der zeitvariablen Mixture-Copulas
% - Param: CopulaParameter
% - weights_tv: Gewichte der Mixture Copula im Zeitverlauf
% - density: Dichten der Copulafunktionen
% - MixDensity: Dichte der dynamischen Mixture Copula
% - PropDensity: Proportional Density der ersten Copula zur Summe der
% CopulaDichten
%
%   Author: Valentin Braun, Martin Grizska
%   Phd Student in finance Goethe Universität, LMU
function [LL_Mix, LL, CopParam_tv, weights_tv, density, MixDensity, PropDensity] =...
    copulaLL_mix_tv_2(tv_faktor, CopParam_1, family, data, P, Q, Density_Func_in, Dynamic);

family = lower(family);
nc = numel(family);
tv_faktor = tv_faktor(:)';
[s1 s2] = size(data);

% Prüfen dass Daten als Zeit x Indices eingeben werden
if s2>s1
    data = data';
    [s1 s2] = size(data);
end

% Prüfen dass Daten [0 1] verteilt sind
if max(max(data))>1 || min(min(data))<0
    error('Daten müssen [0 1] verteilt sein');
end

% Prüfen dass Gewichte in [0,1] liegen
if nc>1
    weights_1 = tv_faktor(end-nc+1 : end)';
    if any(weights_1)<0 || any(weights_1)>1
        error('Gewichte können nur zwischen 0 und 1 liegen!');
    end
end

% Prüfen dass max. 10 Indices verwendet werden
if s2<2 || s2>10
    error('Es können minimal 2 und maximal 10 Zeitreihen in den dynamischen Copulas verwendet werden');
end

% Prüfen dass nur erlaubte Copulafamilien verwendet werden
for i = 1:nc
    if sum(strcmp(family{i}, {'clayton' 'gumbel' 't' 'gaussian'})) ~= 1
        error('angegebene Copulafunktion kann nicht benutzt werden');
    end
end

% -----------------------------------------------------------------------
% Copula Log Likelihood
% -----------------------------------------------------------------------
nf = 0; % Anzahl der Parameter aus der tv_faktor Struktur die schon übergeben wurden
density = zeros(s1,nc); % Platzhalter für Dichten anlegen
for n_Cop = 1:nc
    switch family{n_Cop}
        
        % -----------------------------------------------------------------------
        case 'clayton'
            % Kendalls Tau über Zeithorizont anhand der ARMA Faktoren berechnen
            clear tau archCop_Param
            tau = copulaParam_tv_Patton_2(family{n_Cop},tv_faktor(nf+1:nf+3),CopParam_1(n_Cop),data,'kalibrieren');
            archCop_Param = tau(:)'; % Sicherstellen dass Kendalls Tau als 1xn Vektor übergeben wird
            % Dichte der Clayton Copula berechnen
            density(:,n_Cop) = copulapdfmultivariat(family{n_Cop}, data, archCop_Param(:));
            CopParam_tv{n_Cop} = archCop_Param(:); % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
            LL(n_Cop) = -sum(log(density(:,n_Cop))); % neg. LogLikelihood berechnen
            nf = nf+3; % archimedische Copulas benötigen nach Patton jeweils 1 Konstante, AR, MA Faktoren -> 3 Parameter
            
            % -----------------------------------------------------------------------
        case 'gumbel'
            % Kendalls Tau über Zeithorizont anhand der ARMA Faktoren berechnen
            clear tau archCop_Param
            tau = copulaParam_tv_Patton_2(family{n_Cop},tv_faktor(nf+1:nf+3),CopParam_1(n_Cop),data,'kalibrieren');
            archCop_Param = tau(:)'; % Sicherstellen dass Kendalls Tau als 1xn Vektor übergeben wird
            if ~exist('Density_Func_in') || isempty(Density_Func_in)
                Density_Func = copulapdffunc(family{n_Cop}, s2); % Dichtefunktion herleiten und als Funktion abspeichern. Steigert Zeiteffizienz
            else
                Density_Func = Density_Func_in{n_Cop}; % Dichtefunktion wurde bereits erzeugt und wird hier nur noch ausgelesen
            end
            % Datenmatrix in einzelne Datenvektoren zerlegen
            v = [];
            for i = 1:s2
                var = genvarname('v', who);
                eval([var ' = data(:, i);'])
            end
            % Dichte der entsprechenden archimedischen Copula berechnen
            switch s2
                case 2
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2);
                case 3
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3);
                case 4
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3,v4);
                case 5
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3,v4,v5);
                case 6
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3,v4,v5,v6);
                case 7
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3,v4,v5,v6,v7);
                case 8
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3,v4,v5,v6,v7,v8);
                case 9
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3,v4,v5,v6,v7,v8,v9);
                case 10
                    density(:,n_Cop) = Density_Func(archCop_Param(:), v1,v2,v3,v4,v5,v6,v7,v8,v9,v10);
            end
            
            CopParam_tv{n_Cop} = archCop_Param(:); % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
            LL(n_Cop) = -sum(log(density(:,n_Cop))); % neg. LogLikelihood berechnen
            nf = nf+3; % archimedische Copulas benötigen nach Patton jeweils 1 Konstante, AR, MA Faktoren -> 3 Parameter
            
            % -----------------------------------------------------------------------
        case {'gaussian' 't'}
            y = zeros(s1,s2);
            switch family{n_Cop}
                case 't'
                    if strcmp(Dynamic,'ADCC')
                        DoF = tv_faktor(nf+2+P+Q); % DoF der t-Copula auslesen
                    elseif strcmp(Dynamic,'DCC')
                        DoF = tv_faktor(nf+1+P+Q);
                    elseif strcmp(Dynamic,'GDCC')
                        DoF = tv_faktor(nf+3+P+Q);
                    elseif strcmp(Dynamic,'AGDCC')
                        DoF = tv_faktor(nf+5+P+Q);
                    end
                    for i = 1:s2
                        y(:,i) = tinv(data(:,i),DoF);
                    end
                    
                case 'gaussian'
                    for i = 1:s2
                        y(:,i) = norminv(data(:,i),0,1);
                    end
            end
            if strcmp(Dynamic,'ADCC')
                Corr = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, y, 'kalibrieren');
            elseif strcmp(Dynamic,'DCC')
                Corr = copulaParam_tv_DCC(tv_faktor(nf+1:nf+1*P+Q), P, Q, y, 'kalibrieren');
            elseif strcmp(Dynamic,'GDCC')
                Corr = copulaParam_tv_GDCC(tv_faktor(nf+1:nf+s2*P+s2*Q), P, Q, y, 'kalibrieren');
            elseif strcmp(Dynamic,'AGDCC')
                Corr = copulaParam_tv_AGDCC(tv_faktor(nf+1:nf+2*s2*P+s2*Q), P, Q, y, 'kalibrieren');
            end
            % LL der einzelnen Copula (CL) berechnen
            CL = zeros(1,s1); % Platzhalter Copula Likelihood Werte
            switch family{n_Cop}
                case 't'
                    for i=1:s1
                        CL(i) = gammaln((DoF+s2)/2) + (s2-1)*gammaln(DoF/2) - s2*gammaln((DoF+1)/2) - 0.5*log(det(Corr(:,:,i)));
                        CL(i) = CL(i) - (DoF+s2)/2*log(1+y(i,:)*(Corr(:,:,i))^(-1)*y(i,:)'./(DoF-0));
                        CL(i) = CL(i) + (DoF+1)/2*sum(log(1+(y(i,:).^2/(DoF-0))));
                    end
                    % t-Copulas benötigen nach DCC jeweils 1 Faktor für
                    % ARCH, 1 Faktor für Leverage, 1 Faktor für GARCH; Faktoren entsprechen
                    % denen eines GJR oder EGARCH Modells. Der letzte
                    % Parameter der der tv_Faktoren der t-Copula
                    % repräsentiert den Freiheitsgrad -> 4 Parameter
                    if strcmp(Dynamic,'ADCC')
                        nf = nf+4;
                    elseif strcmp(Dynamic,'DCC')
                        nf = nf+3;
                    elseif strcmp(Dynamic,'GDCC')
                        nf = nf+5;
                    elseif strcmp(Dynamic,'AGDCC')
                        nf = nf+7;
                    end
                case 'gaussian'
                    for i=1:s1
                        CL(i) = -.5*log(det(Corr(:,:,i)));
                        CL(i) = CL(i)-.5*y(i,:)*(inv(Corr(:,:,i))-eye(s2))*y(i,:)';
                    end
                    % Gauss Copulas benötigen nach DCC jeweils 1 Faktor für
                    % ARCH, 1 Faktor für Leverage, 1 Faktor für GARCH; Faktoren entsprechen
                    % denen eines GJR oder EGARCH Modells -> 3 Parameter
                    if strcmp(Dynamic,'ADCC')
                        nf = nf+3;
                    elseif strcmp(Dynamic,'DCC')
                        nf = nf+2;
                    elseif strcmp(Dynamic,'GDCC')
                        nf = nf+4;
                    elseif strcmp(Dynamic,'AGDCC')
                        nf = nf+6;
                    end
            end
            LL(n_Cop) = -sum(CL);
            density(:,n_Cop) = exp(CL)'; % Abspeichern der Dichte für Berechnung der LL der zeitvariablen Mixture Copula
            CopParam_tv{n_Cop} = Corr; % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
    end
end

% Falls Mixture Struktur vorliegt, folgen die Gewichte einem ARMA(1,1) Prozess.
% Berechnen der zeitvariablen Gewichte
if nc > 1
    [weights_tv, PropDensity] = copulamix_Weights_tv_2(tv_faktor(nf+1:end), nc, s1, density);
elseif nc == 1 % einzelne Dynamische Copula
    weights_tv = ones(s1,1);
    PropDensity = [];
end

% LL für die Mixture Copula berechnen. Hierzu werden die Dichten der
% einzelnen Copulas gewichtet, anschließend logarithmiert und aufaddiert
MixDensity = sum(weights_tv .* density, 2);
LL_Mix = -sum(log(MixDensity));
% Sicherstellen dass LL_Mix oder eine einzelne LL nicht NaN, inf oder komplex wird. Dies würde einen
% Fehler in fmincon verursachen
if isnan(LL_Mix) || ~isreal(LL_Mix) || isinf(LL_Mix) || any(isnan(LL)) || any(~isreal(LL)) || any(isinf(LL))
    LL_Mix = 1e6;
end


