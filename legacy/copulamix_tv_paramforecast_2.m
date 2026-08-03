% Version 2:
% Funktion prognostiziert die CopulaParameter der dynamischen Mixture
% Copula. Diese Funktion bezieht sich auf copulafit_mix_tv_2, da hier die
% ARMA Parameter der dynamischen Archimedischen Copulas und des Gewicht
% Prozess dem Stationaritäts- und Invertierbarkeitskriterium folgen. Daher
% können hiermit multiperioden Forecasts generiert werden.
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - tv_faktor: Parameter des ARMA Prozesses des Copulaparameters
% - nc: Anzahl der Copulas
% - data: [0 1] verteilte Zeitreihen
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% - horizon: Zeithorizont der Simulationen. Bsp.: horizon=10 -> die
% CopulaParameter der dynamischen Mixture Copula werden für den Zeitpunt
% t+10 prognostiziert.
% P,Q werden nur für die elliptischen Copulas übergeben!
%
% OUTPUT:
% - CopParam_pred: CopulaParameter zum Zeitpunkt [t+1,...,t+horizon]
% - Wpred: prognostizierte Gewichte
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function [CopParam_pred, Wpred] = copulamix_tv_paramforecast_2(tv_faktor, nc, family, data, P, Q, horizon, Dynamic);

[s1 s2] = size(data);

%--------------------------------------------------------------------------
% Rekursives Prognostizieren der CopulaParameter
%--------------------------------------------------------------------------
% Erwartungswert wird auf 0 gesetzt. Da data als [0 1] verteilte Variable
% übergeben wird, muss der Erwartungswert auch in diese Skala transformiert
% werden. Dies ergibt einen exp_ret von 0.5
exp_ret = 0.5 .* ones(horizon,s2); % CDF Erwartungswert
nf = 0; % Anzahl der bereits ausgelesen zeitvariablen Paramter
for j = 1:nc
    switch family{j}
        %         case {'clayton' 'gumbel'}
        %             CopParam_1(j) = copulafitmultivariat(family{j}, data); % Statischer Copulaparameter der archimedischen Copula als Startwert
        %             tau_pred = copulaParam_tv_Patton(family{j},tv_faktor(nf+1:nf+3),CopParam_1(j),data,'vorhersage',horizon);
        %             if strcmp(family{j}, 'clayton')
        %                 CopParam_pred{j} = 2*tau_pred./(1-tau_pred); % Kendalls Tau in Clayton Parameter umrechnen
        %             elseif strcmp(family{j}, 'gumbel')
        %                 CopParam_pred{j} = 1 ./(1-tau_pred) + 1e-3; % Kendalls Tau in Gumbel Parameter umrechnen & sicherstellen dass CopParam>0
        %             end
        %             for i = 1:horizon
        %                 % Out-of-Sample Dichte mit prognostizierten CopulaParameter berechnen
        %                 density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), CopParam_pred{j}(i));
        %             end
        %             nf = nf+3;
        case {'clayton' 'gumbel'}
            CopParam_1(j) = copulafitmultivariat(family{j}, data); % Statischer Copulaparameter der archimedischen Copula als Startwert
            % ARMA Parameter des dynamischen Copulaparameterprozess auslesen
            Kac = tv_faktor(nf+1);
            ARac = tv_faktor(nf+2);
            MAac = tv_faktor(nf+3);
            % historischen Verlauf des CopulaParameters (CPH) via des dynamischen
            % Prozess berechnen. Die Funktion copulaParam_tv_Patton_2 gibt
            % dabei schon die Copulaparameter für die Archimedischen
            % Copulas aus; NICHT Kendalls Tau! Diese Copulaparameter müssen
            % dann erst rücktransformiert werden damit sie im ARMA Prozess
            % implementiert werden können.
            % GD ist die absolute Gesamtdistanz der Indizes an jedem
            % Zeitpunkt geteilt durch die Laganzahl
            [CPHT, GD] = copulaParam_tv_Patton_2(family{j},[Kac;ARac;MAac],CopParam_1(j),data,'kalibrieren');
            switch family{j}
                case 'clayton'
                    CPH = log(CPHT - 1e-6); % Rücktransformation
                case 'gumbel'
                    CPH = log(CPHT - 1 - 1e-6); % Rücktransformation
            end
            GDV = var(GD); % Varianz des Störterms
            % multistep Forecast des Erwartungswerts eines ARMA(1,1) Prozess.
            % Walter (2004), Applied Econometric Time Series, S. 104, Formel 2.64
            % multistep Forecast der Varianz eines ARMA(1,1) Prozess;
            % Tsay, Analysis of Financial Time Series, S. 48-56 & Leyold, ARMA
            % Die Copulaparameter werden hier OHNE Transformation in
            % copulaParam_tv_Patton_2 übergeben.
            for i = 1:horizon % Forecast für die Zeitpunkte ab (t+1)
                Farma(i) = (Kac./(1 - ARac)) .* (1 - ARac.^i) + ...
                    (MAac .* ARac.^(i-1) .* GD(end)) + (ARac.^i) .* CPH(end); % ARMA(1,1) Forecast
                %                 VP(i) = (MAac^2 *(1-ARac^(2*i)))./(1-ARac^2); % Hafner & Manner: DYNAMIC STOCHASTIC COPULA MODELS: ESTIMATION, INFERENCE AND APPLICATIONS
                if i == 1
                    VP(i) = GDV;
                else
                    VP(i) = (1+MAac.^2).*GDV; % Varianzforecast
                end
            end
            % Die ARMA Prognose erzeugt noch keine Copulaparameter. Die
            % erzeugten Werte müssen wie in copulaParam_tv_Patton_2 noch
            % entsprechend transformiert werden. Die
            % Exponentialtransformation wird entsprechend Hafner & Manner:
            % DYNAMIC STOCHASTIC COPULA MODELS: ESTIMATION, INFERENCE AND
            % APPLICATIONS ausgeführt. Hier wird noch die prognostizierte Varinaz des
            % Störterms auf die Parameterprognose addiert.
            switch family{j}
                case 'clayton'
                    CopParam_pred{j} = exp(Farma + VP./2) + 1e-6; % Sicherstellen dass der Parameter > 0
                case 'gumbel'
                    CopParam_pred{j} = exp(Farma + VP./2) + 1 + 1e-6; % Sicherstellen dass der Parameter > 1
            end
            nf = nf+3;
            
        case 't'
            data_inv = zeros(s1,s2);
            DoF = tv_faktor(nf+2+P+Q); % DoF der t-Copula auslesen
            CopParam_1(j) = 0; % Platzhalter anlegen
            for i = 1:s2
                data_inv(:,i) = tinv(data(:,i),DoF(j)); % Invertieren der [0 1] verteilten Daten
            end
            Rbar = corr(data_inv);
            % Prognostizieren der Korrelationsmatrix zum Zeitpunkt (t+1)
            % nach Cappiello et al. (2006)
            R_pred = zeros(s2, s2, horizon);
            if strcmp(Dynamic,'ADCC')
                R_pred(:,:,1) = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'DCC')
                R_pred(:,:,1) = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'GDCC')
                R_pred(:,:,1) = copulaParam_tv_GDCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'AGDCC')
                R_pred(:,:,1) = copulaParam_tv_AGDCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            end
            % Prognostizieren der Korrelationsmatrizen für die Zeitpunkte
            % [t+2,...,t+n]. Dabei wird auf die Formulierung des TGARCH
            % Modells in Zivot (2008) und DCC Modell von Engle & Sheppard
            % (2001) zurückgegriffen
            sumTerm = zeros(s2, s2, horizon);
            for i = 0:horizon-2
                sumTerm(:,:,i+2) = sumTerm(:,:,i+1) + ((1 - sum(tv_faktor(1:P)) - tv_faktor(1+2*P:2*P+Q) - tv_faktor(1+P:2*P)./2) ...
                    .* Rbar .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^i);
                % Für (t+1) prognostizierte Kovarianzmatrix wird mit den
                % Koeffizienten gewichtet und auf sumTerm aufaddiert
                R_pred(:,:,i+2) = sumTerm(:,:,i+2) + R_pred(:,:,1) ...
                    .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^(i+1);
                % Sicherstellen dass R_pred pos. definit ist
                R_pred(:,:,i+2) = R_pred(:,:,i+2) .* (eye(s2)==0) + eye(s2);
            end
            for i = 1:horizon
                % Out-of-Sample Dichte mit prognostizierten CopulaParametern berechnen
                density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), R_pred(:,:,i), DoF);
            end
            CopParam_pred{j}{1} = R_pred; % Übergeben der Korrelationsmatrizen
            CopParam_pred{j}{2} = DoF; % Übergeben des Freiheitsgrades
            nf = nf+4;
            
        case 'gaussian'
%             -------------------------------------------------------------
%             erzeuge stdresid aus den univariate gesch�tzten parametern
            [t,k]=size(data);
            index=1;
            H=zeros(size(data));
            stdEstimate =  std(data,1);
            P=dccP;
            Q=dccQ;
            m=max(max(archP),max(garchQ));            
            % for of univariate (garch)parameters is [omega; alpha; talpha; beta; nu; skew-t];
            % if estimated with skew-t add two parameters; if estimated with student-t
            % or GED add one parameter; if estimated with normal add 0 parameters
            gamma=zeros(k,1);
            for i=1:k
                if errortype(i) ~= 1 && errortype(i) ~= 4
                    gamma(i)=1;
                elseif errortype(i) == 4
                    gamma(i) = 2;
                else
                    gamma(i)=0;
                end
            end
            for i=1:k
                %     select univariate GARCH-params + distribution params
                univariateparams=params(index:index+archP+o(i)+garchQ+gamma);
                if garchtype(i)==0;
                    H(:,i) = egarchcore(data(:,i), univariateparams, stdEstimate(:,i), archP, o(i), garchQ, m , t);
                else
                    %         set leverage term to zero if model contains not leverage but
                    %         power-parameter
                    if garchtype(i) == 4
                        o2(i) = 0;
                    elseif garchtype(i) == 6
                        %     special case: APGARCH contains leverage and power-parameter
                        o2(i) = 1;
                    else
                        o2(i) = o(i);
                    end
                    [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparams, data(:,i), archP, o2(i), garchQ, garchtype(i), errortype(i), stdEstimate(i));
                end
                index=index+archP+garchQ+o(i)+gamma(i)+1;
            end            
            stdresid=data./sqrt(H);            
            U=zeros(t,k);
            for i=1:k
                if strcmp(GARCHOutput{i}.dist,'GAUSS')
                    U(:,i) = normcdf(stdresid(:,i));
                elseif strcmp(GARCHOutput{i}.dist,'STUDENTST')
                    U(:,i) = tcdf(stdresid(:,i), GARCHOutput{i}.Params(end));
                elseif strcmp(GARCHOutput{i}.dist,'GED')
                    U(:,i) = gedcdf(stdresid(:,i), GARCHOutput{i}.Params(end));
                elseif strcmp(GARCHOutput{i}.dist,'SKEWT')
                    U(:,i) = skewtdis_cdf(stdresid(:,i),GARCHOutput{i}.Params(end-1),GARCHOutput{i}.Params(end));
                end
            end
            
            data_inv = zeros(s1,s2);
            CopParam_1(j) = 0; % Platzhalter anlegen
            for i = 1:s2
                data_inv(:,i) = norminv(U(:,i),0,1); % Invertieren der [0 1] verteilten Daten
            end
            % Prognostizieren der Korrelationsmatrizen über die Prognosezeitpunkte
            %             R_pred = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', horizon);
            Rbar = corr(data_inv);
            % Prognostizieren der Korrelationsmatrix zum Zeitpunkt (t+1)
            % nach Cappiello et al. (2006)
            R_pred = zeros(s2, s2, horizon);
            if strcmp(Dynamic,'ADCC')
                R_pred(:,:,1) = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'DCC')
                R_pred(:,:,1) = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'GDCC')
                R_pred(:,:,1) = copulaParam_tv_GDCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'AGDCC')
                R_pred(:,:,1) = copulaParam_tv_AGDCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
            end
            % Prognostizieren der Korrelationsmatrizen für die Zeitpunkte
            % [t+2,...,t+n]. Dabei wird auf die Formulierung des TGARCH
            % Modells in Zivot (2008) und DCC Modell von Engle & Sheppard
            % (2001) zurückgegriffen
            sumTerm = zeros(s2, s2, horizon);
            for i = 0:horizon-2
                sumTerm(:,:,i+2) = sumTerm(:,:,i+1) + ((1 - sum(tv_faktor(1:P)) - tv_faktor(1+2*P:2*P+Q) - tv_faktor(1+P:2*P)./2) ...
                    .* Rbar .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^i);
                % Für (t+1) prognostizierte Kovarianzmatrix wird mit den
                % Koeffizienten gewichtet und auf sumTerm aufaddiert
                R_pred(:,:,i+2) = sumTerm(:,:,i+2) + R_pred(:,:,1) ...
                    .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^(i+1);
                % Sicherstellen dass R_pred pos. definit ist
                R_pred(:,:,i+2) = R_pred(:,:,i+2) .* (eye(s2)==0) + eye(s2);
            end
            for i = 1:horizon
                % Out-of-Sample Dichte mit prognostizierten CopulaParameter berechnen
                density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), R_pred(:,:,i));
            end
            CopParam_pred{j} = R_pred; % Übergeben der Korrelationsmatrizen
            nf = nf+3;
    end
end

%--------------------------------------------------------------------------
% Prognostizieren der CopulaGewichte
% siehe Tsay, Analysis of Financial Time Series, S. 49-56
%--------------------------------------------------------------------------
% Berechnen der InSample Dichten via LL Funktion
% WH: Historic Weights, DH: historische Dichten der Copulas in der Mixture,
% PD: Proportional Density der ersten Copula zur Summe der Copuladichten
[d1, d2, d3, WH, DH, d4, PD] = copulaLL_mix_tv_2(tv_faktor, CopParam_1, family, data, P, Q);
clear d1 d2 d3 d4

% multistep Forecast des Erwartungswerts eines ARMA(1,1) Prozess.
% Walter (2004), Applied Econometric Time Series, S. 104, Formel 2.64
% multistep Forecast der Varianz eines ARMA(1,1) Prozess;
% Tsay, Analysis of Financial Time Series, S. 48-56 & Leyold, ARMA
% Die Parameter des ARMA Prozess der Gewichte werden NICHT transformiert.
% Daher ist hier auch keine Rücktransformation oder Berechnung der Varianz
% der ARMA Forecasts nötig.
if nc > 1
    Kw = tv_faktor(nf+1); % Konstante des ARMA(1,1) Prozess
    ARw = tv_faktor(nf+2:nf+nc); % AR-Faktoren
    MAw = tv_faktor(nf+1+nc:nf+1+2*(nc-1)); % MA-Faktoren
    for i = 1:horizon % Forecast für die Zeitpunkte ab (t+1)
        W1pred(i,1) = (Kw./(1 - ARw)) .* (1 - ARw.^i) + ...
            (MAw .* ARw.^(i-1) .* WH(end,1)) + (ARw.^i) .* PD(end); % ARMA(1,1) Forecast
    end
    Wpred = [W1pred, 1-W1pred];
else
    Wpred = ones(horizon, 1);
end









