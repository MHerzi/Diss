% Funktion prognostiziert die CopulaParameter der dynamischen Mixture
% Copula.
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
% - weights_pred: prognostizierte Gewichte
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function [CopParam_pred, weights_pred] = copulamix_tv_paramforecast_grm(tv_faktor, nc, family, data, P, Q, horizon, Dynamic);

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
        case {'clayton' 'gumbel'}
            CopParam_1(j) = copulafitmultivariat(family{j}, data); % Statischer Copulaparameter der archimedischen Copula als Startwert
            tau_pred = copulaParam_tv_Patton_grm2_original(family{j},tv_faktor(nf+1:nf+3),CopParam_1(j),data,'vorhersage',horizon);
            if strcmp(family{j}, 'clayton')
                CopParam_pred{j} = 2*tau_pred./(1-tau_pred); % Kendalls Tau in Clayton Parameter umrechnen
            elseif strcmp(family{j}, 'gumbel')
                CopParam_pred{j} = 1 ./(1-tau_pred) + 1e-3; % Kendalls Tau in Gumbel Parameter umrechnen & sicherstellen dass CopParam>0
            end
            for i = 1:horizon
                % Out-of-Sample Dichte mit prognostizierten CopulaParameter berechnen
                density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), CopParam_pred{j}(i));
            end
            nf = nf+3;
            
        case 't'
            data_inv = zeros(s1,s2);
            if strcmp(Dynamic,'ADCC')
                DoF = tv_faktor(nf+2+P+Q); % DoF der t-Copula auslesen
            elseif strcmp(Dynamic,'DCC')
                DoF = tv_faktor(nf+1+P+Q);
            elseif strcmp(Dynamic,'GDCC')
                DoF = tv_faktor(nf+1+(P+Q)*s2);
            elseif strcmp(Dynamic,'AGDCC')
                DoF = tv_faktor(nf+1+(2*P+Q)*s2);
            end
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
                R_pred(:,:,1) = copulaParam_tv_DCC(tv_faktor(nf+1:nf+P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'GDCC')
                R_pred(:,:,1) = copulaParam_tv_GDCC(tv_faktor(nf+1:nf+(P+Q)*s2), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'AGDCC')
                R_pred(:,:,1) = copulaParam_tv_GDCC(tv_faktor(nf+1:nf+(2*P+Q)*s2), P, Q, data_inv, 'vorhersage', 1);
            end
            % Prognostizieren der Korrelationsmatrizen für die Zeitpunkte
            % [t+2,...,t+n]. Dabei wird auf die Formulierung des TGARCH
            % Modells in Zivot (2008) und DCC Modell von Engle & Sheppard
            % (2001) zurückgegriffen
            sumTerm = zeros(s2, s2, horizon);
            for i = 0:horizon-2
                if strcmp(Dynamic,'ADCC')
                    sumTerm(:,:,i+2) = sumTerm(:,:,i+1) + ((1 - sum(tv_faktor(1:P)) - tv_faktor(1+2*P:2*P+Q) - tv_faktor(1+P:2*P)./2) ...
                        .* Rbar .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^i);
                elseif strcmp(Dynamic,'DCC')
                    sumTerm(:,:,i+2) = sumTerm(:,:,i+1) + ((1 - sum(tv_faktor(1:P)) - tv_faktor(1+P:P+Q)) ...
                        .* Rbar .* (sum(tv_faktor(1:P)) + tv_faktor(1+P:+Q)).^i);
                end
                % Für (t+1) prognostizierte Kovarianzmatrix wird mit den
                % Koeffizienten gewichtet und auf sumTerm aufaddiert
                if strcmp(Dynamic,'ADCC')
                    R_pred(:,:,i+2) = sumTerm(:,:,i+2) + R_pred(:,:,1) ...
                        .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^(i+1);
                elseif strcmp(Dynamic,'DCC')
                    R_pred(:,:,i+2) = sumTerm(:,:,i+2) + R_pred(:,:,1) ...
                        .* (sum(tv_faktor(1:P)) + tv_faktor(1+P:P+Q)).^(i+1);
                end
                % Sicherstellen dass R_pred pos. definit ist
                R_pred(:,:,i+2) = R_pred(:,:,i+2) .* (eye(s2)==0) + eye(s2);
            end
            for i = 1:horizon
                % Out-of-Sample Dichte mit prognostizierten CopulaParametern berechnen
                density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), R_pred(:,:,i), DoF);
            end
            CopParam_pred{j}{1} = R_pred; % Übergeben der Korrelationsmatrizen
            CopParam_pred{j}{2} = DoF; % Übergeben des Freiheitsgrades
            if strcmp(Dynamic,'ADCC')
                nf = nf+4;
            elseif strcmp(Dynamic,'DCC');
                nf = nf+3;
            end
            
        case 'gaussian'
            data_inv = zeros(s1,s2);
            CopParam_1(j) = 0; % Platzhalter anlegen
            for i = 1:s2
                data_inv(:,i) = norminv(data(:,i),0,1); % Invertieren der [0 1] verteilten Daten
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
                R_pred(:,:,1) = copulaParam_tv_DCC(tv_faktor(nf+1:nf+P+Q), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'GDCC')
                R_pred(:,:,1) = copulaParam_tv_GDCC(tv_faktor(nf+1:nf+(P+Q)*2), P, Q, data_inv, 'vorhersage', 1);
            elseif strcmp(Dynamic,'AGDCC')
                R_pred(:,:,1) = copulaParam_tv_AGDCC(tv_faktor(nf+1:nf+(2*P+Q)*2), P, Q, data_inv, 'vorhersage', 1);
            end
    end
    % Prognostizieren der Korrelationsmatrizen für die Zeitpunkte
    % [t+2,...,t+n]. Dabei wird auf die Formulierung des TGARCH
    % Modells in Zivot (2008) und DCC Modell von Engle & Sheppard
    % (2001) zurückgegriffen
    sumTerm = zeros(s2, s2, horizon);
    for i = 0:horizon-2
        if strcmp(Dynamic,'ADCC')
            sumTerm(:,:,i+2) = sumTerm(:,:,i+1) + ((1 - sum(tv_faktor(1:P)) - tv_faktor(1+2*P:2*P+Q) - tv_faktor(1+P:2*P)./2) ...
                .* Rbar .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^i);
        elseif strcmp(Dynamic,'DCC')
            sumTerm(:,:,i+2) = sumTerm(:,:,i+1) + ((1 - sum(tv_faktor(1:P)) - tv_faktor(1+P:P+Q)) ...
                .* Rbar .* (sum(tv_faktor(1:P)) + tv_faktor(1+P:P+Q)).^i);
        end
        % Für (t+1) prognostizierte Kovarianzmatrix wird mit den
        % Koeffizienten gewichtet und auf sumTerm aufaddiert
        if strcmp(Dynamic,'ADCC')
            R_pred(:,:,i+2) = sumTerm(:,:,i+2) + R_pred(:,:,1) ...
                .* (sum(tv_faktor(1:P)) + tv_faktor(1+2*P:2*P+Q) + tv_faktor(1+P:2*P)./2).^(i+1);
        elseif strcmp(Dynamic,'DCC')
            R_pred(:,:,i+2) = sumTerm(:,:,i+2) + R_pred(:,:,1) ...
                .* (sum(tv_faktor(1:P)) + tv_faktor(1+P:P+Q)).^(i+1);
        end
        % Sicherstellen dass R_pred pos. definit ist
        R_pred(:,:,i+2) = R_pred(:,:,i+2) .* (eye(s2)==0) + eye(s2);
    end
    for i = 1:horizon
        % Out-of-Sample Dichte mit prognostizierten CopulaParameter berechnen
        density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), R_pred(:,:,i));
    end
    CopParam_pred{j} = R_pred; % Übergeben der Korrelationsmatrizen
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

%--------------------------------------------------------------------------
% Rekursives Prognostizieren der CopulaGewichte
%--------------------------------------------------------------------------
weights_1 = ones(nc,1)./nc; % Startgewichte gleichgewichtet festlegen (wie auch bei der Parameterschätzung)
% Berechnen der InSample Dichten via LL Funktion
[d1, d2, d3, weights_hist, density_insample] = copulaLL_mix_tv_grm(tv_faktor, CopParam_1, weights_1, family, data, P, Q, Dynamic);
clear d1 d2 d3

% Die Gewichte der Mixture Copula ergeben sich aus den vorangegangenen
% Gewichten und dem Verhältniss der Dichten der Copulas. Daher werden hier
% die InSample & OutSample Dichten aneinander gehängt. Dann können die
% Mixture Gewichte rekursiv ermittelt werden.
density = [density_insample; density_outsample];
dL = s1+horizon; % Länge des gesamten Density Vektors
weights_rekursiv = copulamix_Weights_tv_grm(tv_faktor(nf+1:nf+2*(nc-1)), nc, dL, weights_1, density, Dynamic);
weights_pred = weights_rekursiv(s1+1 : s1+horizon, :); % Prognosegewichte auslesen

% [weights_tv, MD] = copulamix_Weights_tv(tv_faktor(nf+1:nf+2*(nc-1)), nc, s1, weights_1, density_insample);
% % multistep Forecast mittels ARMA Prozess. Walter (2004),
% % Applied Econometric Time Series, S. 104, Formel 2.64
% weights_faktor_trans = 1./(1+exp(-tv_faktor(nf+1:nf+2*(nc-1)))); % Logistic Transformation nach Patton (2006)
% for i = 1:horizon % Forecast für die Zeitpunkte ab (t+1)
%     Weights_pred(i) = (weights_faktor_trans(2) .* weights_faktor_trans(1).^(i-1) .* MD) +...
%         (weights_faktor_trans(1).^i) .* weights_hist(end);
% end









