% LL Function der dynamischen Mixture Copulas. 
% Zulässige Copula Familien: t, Gaussian, Clayton, Gumbel
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - CopParam_1: bei archimedischen Copulas: erster CopulaParameter im Zeitablauf
%                            bei elliptischen Copulas wird dieser auf 0
%                            gesetzt
% - weights_1: Startwerte der Gewichte
% - tv_faktor: Parameter des ARMA Prozesses des Copulaparameters
% - data: [0 1] verteilte Zeitreihen
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% P,Q werden nur für die elliptischen Copulas übergeben!
%
% OUTPUT:
% - LL: LogLikelihood Wert der einzelnen zeitvariablen Copulas
% - LL_Mix: LogLikelihood Wert der zeitvariablen Mixture-Copulas
% - Param: CopulaParameter
% - weights_tv: Gewichte der Mixture Copula im Zeitverlauf
% - density: Dichten der Copulafunktionen
% - likelihoods_mix: Vektor mit den likelihoods f�r jeden Zeitpunkt t der
%                    Mixture Copuly
% - likelihoods: Vektor mit den likelihoods f�r jeden Zeitpunkt t der
%                einzelnen zeitvariablen Copulas
%
%   Author: Valentin Braun, Martin Grizska
%   Phd Student in finance Goethe Universität, LMU
function [LL_Mix, LL, CopParam_tv, weights_tv, density, MixDensity, PropDensity] = copulaLL_mix_tv_grm2(tv_faktor, CopParam_1, family, data, P, Q, Dynamic, Density_Func_in, epsilon);

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
if s2<2 || s2>14
    error('Es können minimal 2 und maximal 10 Zeitreihen in den dynamischen Copulas verwendet werden');
end

% Prüfen dass nur erlaubte Copulafamilien verwendet werden
for i = 1:nc
    if sum(strcmp(family{i}, {'clayton' 'gumbel' 't' 'gaussian' 'rotclayton'})) ~= 1
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
        case {'clayton','rotclayton'}
            if strcmp(family{n_Cop},'rotclayton')
                data = 1 - data; %drehe f�r rotated clayton den datensatz
            end
            % Kendalls Tau über Zeithorizont anhand der ARMA Faktoren berechnen
            clear tau archCop_Param
            if s2==2 % f�r zwei Zeitreihen greife auf die Originalspezifikation van Patton zur�ck
                tau = copulaParam_tv_Patton_grm2_original(family{n_Cop},tv_faktor(nf+1:nf+3),CopParam_1(n_Cop),data,'kalibrieren');
            else
                tau = copulaParam_tv_Patton_grm2(family{n_Cop},tv_faktor(nf+1:nf+3),CopParam_1(n_Cop),data,'kalibrieren');
            end
            tau = tau(:)'; % Sicherstellen dass Kendalls Tau als 1xn Vektor übergeben wird
            archCop_Param = tau(:)'; % Sicherstellen dass Kendalls Tau als 1xn Vektor übergeben wird
            % Dichte der Clayton Copula berechnen
            density(:,n_Cop) = copulapdfmultivariat_grm(family{n_Cop}, data, archCop_Param(:)); 
            CopParam_tv{n_Cop} = archCop_Param(:); % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
            LL(n_Cop) = -sum(log(density(:,n_Cop))); % neg. LogLikelihood berechnen
            nf = nf+3; % archimedische Copulas benötigen nach Patton jeweils 1 Konstante, AR, MA Faktoren -> 3 Parameter
        
% -----------------------------------------------------------------------        
        case 'gumbel'
            % Kendalls Tau über Zeithorizont anhand der ARMA Faktoren berechnen
            clear tau archCop_Param 
            if s2==2 % f�r zwei Zeitreihen nehme die Original-Spezifikation von PAtton
                tau = copulaParam_tv_Patton_grm2_original(family{n_Cop},tv_faktor(nf+1:nf+3),CopParam_1(n_Cop),data,'kalibrieren');
            else
                tau = copulaParam_tv_Patton_grm2(family{n_Cop},tv_faktor(nf+1:nf+3),CopParam_1(n_Cop),data,'kalibrieren');
            end
            archCop_Param = tau(:)'; % Sicherstellen dass Kendalls Tau als 1xn Vektor übergeben wird
            if ~exist('Density_Func_in') || isempty(Density_Func_in)
                Density_Func = copulapdffunc_grm(family{n_Cop}, s2); % Dichtefunktion herleiten und als Funktion abspeichern. Steigert Zeiteffizienz 
            else
                Density_Func = Density_Func_in{n_Cop}; % Dichtefunktion wurde bereits erzeugt und wird hier nur noch ausgelesen
            end
            % Datenmatrix in einzelne Datenvektoren zerlegen
            v = [];
            for i = 1:s2
               var = genvarname('v', who);
               eval([var ' = data(:, i);'])
            end
            for i = 1:s1
                % Dichte der entsprechenden archimedischen Copula berechnen
                switch s2
                    case 2
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i));
                    case 3
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i));
                    case 4
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i));
                    case 5
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i));
                    case 6
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i));
                    case 7
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i));
                    case 8
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i));
                    case 9
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                    case 10
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v10(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i)); %!!! Matlab ordnet die Daten v1 v10 v2...
                    case 11
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v10(i),v11(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));%!!! Matlab ordnet die Daten v1 v10 v11 v2...
                    case 12
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));%!!! Matlab ordnet die Daten v1 v10 v11 v12 v2...
                    case 13
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v13(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));%!!! Matlab ordnet die Daten v1 v10 v11 v12 v13 v2...
                    case 14
                        density(i,n_Cop) = Density_Func(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v13(i),v14(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));%!!! Matlab ordnet die Daten v1 v10 v11 v12 v13 v14 v2...
                end
            end
            CopParam_tv{n_Cop} = archCop_Param(:); % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
            LL(n_Cop) = -sum(log(density(:,n_Cop))); % neg. LogLikelihood berechnen
            likelihoods(:,n_Cop) = log(density(:,n_Cop));
            nf = nf+3; % archimedische Copulas benötigen nach Patton jeweils 1 Konstante, AR, MA Faktoren -> 3 Parameter
        
% -----------------------------------------------------------------------        
        case {'gaussian' 't'}
            y = zeros(s1,s2);
            switch family{n_Cop}
                case 't'
                    if strcmp(Dynamic,'ADCC')
                        DoF = tv_faktor(nf+1+2*P+Q); % DoF der t-Copula auslesen
                    elseif strcmp(Dynamic,'DCC')
                        DoF = tv_faktor(nf+1+P+Q);% DoF der t-Copula auslesen
                    elseif strcmp(Dynamic,'GDCC')
                        DoF = tv_faktor(nf+1+P*s2+Q*s2);
                    elseif strcmp(Dynamic,'AGDCC')
                        DoF = tv_faktor(nf+1+2*P*s2+Q*s2);
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
                Corr = copulaParam_tv_DCC(tv_faktor(nf+1:nf+P+Q), P, Q, y, 'kalibrieren');
            elseif strcmp(Dynamic,'GDCC')
                Corr = copulaParam_tv_GDCC(tv_faktor(nf+1:nf+P*s2+Q*s2), P, Q, y, 'kalibrieren');
            elseif strcmp(Dynamic,'AGDCC')
                Corr = copulaParam_tv_AGDCC(tv_faktor(nf+1:nf+2*P*s2+Q*s2), P, Q, y, 'kalibrieren');
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
                    if nc > 1
                        if strcmp(Dynamic,'ADCC')
                            nf = nf+4;
                        elseif strcmp(Dynamic,'DCC')
                            nf = nf+3;
                        elseif strcmp(Dynamic,'GDCC')
                            nf = nf+s2*2+1; % s2*2 Parameter f�r ARCH und GARCH pro Zeitreihe, 1 Parameter f�r den Freiheitsgrad, bei (1,1)-Modellen
                        elseif strcmp(Dynamic,'AGDCC')
                            nf = nf+s2*3+1; % s2*2 Parameter f�r ARCH und GARCH pro Zeitreihe, 1 Parameter f�r den Freiheitsgrad, bei (1,1)-Modellen
                        end
                    end
                case 'gaussian'
                    for i=1:s1
                        CL(i) = -.5*log(det(Corr(:,:,i)));
                        CL(i) = CL(i)-.5*y(i,:)*(inv(Corr(:,:,i))-eye(s2))*y(i,:)';
                    end
                    % Gauss Copulas benötigen nach DCC jeweils 1 Faktor für
                    % ARCH, 1 Faktor für Leverage, 1 Faktor für GARCH; Faktoren entsprechen 
                    % denen eines GJR oder EGARCH Modells -> 3 Parameter
                    if nc > 1
                        if strcmp(Dynamic,'ADCC')
                            nf = nf+3;
                        elseif strcmp(Dynamic,'DCC')
                            nf = nf+2;
                        elseif strcmp(Dynamic,'GDCC')
                            nf = nf+s2*2;
                        elseif strcmp(Dynamic,'AGDCC')
                            nf = nf+s2*3;
                        end
                    end
            end
            LL(n_Cop) = -sum(CL);
            density(:,n_Cop) = exp(CL)'; % Abspeichern der Dichte für Berechnung der LL der zeitvariablen Mixture Copula
            likelihoods(:,n_Cop) = CL;
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
        LL_Mix = 1e10;
end


