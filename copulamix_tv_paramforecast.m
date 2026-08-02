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
function [CopParam_pred, weights_pred] = copulamix_tv_paramforecast(tv_faktor, nc, family, data, P, Q, horizon);

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
            tau_pred = copulaParam_tv_Patton(family{j},tv_faktor(nf+1:nf+3),CopParam_1(j),data,'vorhersage',horizon); 
            if strcmp(family{j}, 'clayton')
                CopParam_pred{j} = 2*tau_pred./(1-tau_pred); % Kendalls Tau in Clayton Parameter umrechnen
            elseif strcmp(family{j}, 'gumbel')
                CopParam_pred{j} = 1 ./(1-tau) + 1e-3; % Kendalls Tau in Gumbel Parameter umrechnen & sicherstellen dass CopParam>0
            end
            for i = 1:horizon
                % Out-of-Sample Dichte mit prognostizierten CopulaParameter berechnen
                density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), CopParam_pred{j}(i)); 
            end
            nf = nf+3;
%         case {'clayton' 'gumbel'}
%             CopParam_1(j) = copulafitmultivariat(family{j}, data); % Statischer Copulaparameter der archimedischen Copula als Startwert
%             % tau & GesamtDistanz (GD) über 10 Datenpunkte der [0 1] verteilten Daten zum Zeitpunkt (t0)
%             [tau_hist, GD, MALags] = copulaParam_tv_Patton(family{j},tv_faktor(nf+1:nf+3),CopParam_1(j),data,'kalibrieren'); 
%             if strcmp(family{j},'clayton')
%                 archCop_Param_hist = 2.*tau_hist./(1-tau_hist); % Kendalls Tau in Clayton Parameter umrechnen
%             elseif strcmp(family{j},'gumbel')
%                 archCop_Param_hist = 1 ./(1-tau_hist) + 1e-3; % Kendalls Tau in Gumbel Parameter umrechnen
%             end
%             % multistep Forecast mittels ARMA Prozess. Walter (2004),
%             % Applied Econometric Time Series, S. 104, Formel 2.64
%             tv_faktor_trans = 1./(1+exp(-tv_faktor(nf+1:nf+3))); % Logistic Transformation nach Patton (2006)
%             for i = 1:horizon % Forecast für die Zeitpunkte ab (t+1)
%                 CopParam_pred{j}(i) = (tv_faktor_trans(1)./(1 - tv_faktor_trans(2))) .* (1 - tv_faktor_trans(2).^i) + ...
%                     (tv_faktor_trans(3) .* tv_faktor_trans(2).^(i-1) .* GD(end)) + (tv_faktor_trans(2).^i) .* archCop_Param_hist(end);
%             end
%             for i = 1:horizon
%                 % Out-of-Sample Dichte mit prognostizierten CopulaParameter berechnen
%                 density_outsample(i,j) = copulapdfmultivariat(family{j}, exp_ret(i,:), CopParam_pred{j}(i)); 
%             end
%             nf = nf+3;
            
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
            R_pred(:,:,1) = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
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
            R_pred(:,:,1) = copulaParam_tv_ADCC(tv_faktor(nf+1:nf+2*P+Q), P, Q, data_inv, 'vorhersage', 1);
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
% Rekursives Prognostizieren der CopulaGewichte
%--------------------------------------------------------------------------
weights_1 = ones(nc,1)./nc; % Startgewichte gleichgewichtet festlegen (wie auch bei der Parameterschätzung)
% Berechnen der InSample Dichten via LL Funktion
[d1, d2, d3, weights_hist, density_insample] = copulaLL_mix_tv(tv_faktor, CopParam_1, weights_1, family, data, P, Q);
clear d1 d2 d3 

% Die Gewichte der Mixture Copula ergeben sich aus den vorangegangenen
% Gewichten und dem Verhältniss der Dichten der Copulas. Daher werden hier
% die InSample & OutSample Dichten aneinander gehängt. Dann können die
% Mixture Gewichte rekursiv ermittelt werden. 
density = [density_insample; density_outsample];
dL = s1+horizon; % Länge des gesamten Density Vektors
weights_rekursiv = copulamix_Weights_tv(tv_faktor(nf+1:nf+2*(nc-1)), nc, dL, weights_1, density);
weights_pred = weights_rekursiv(s1+1 : s1+horizon, :); % Prognosegewichte auslesen

% [weights_tv, MD] = copulamix_Weights_tv(tv_faktor(nf+1:nf+2*(nc-1)), nc, s1, weights_1, density_insample);
% % multistep Forecast mittels ARMA Prozess. Walter (2004),
% % Applied Econometric Time Series, S. 104, Formel 2.64
% weights_faktor_trans = 1./(1+exp(-tv_faktor(nf+1:nf+2*(nc-1)))); % Logistic Transformation nach Patton (2006)
% for i = 1:horizon % Forecast für die Zeitpunkte ab (t+1)
%     Weights_pred(i) = (weights_faktor_trans(2) .* weights_faktor_trans(1).^(i-1) .* MD) +... 
%         (weights_faktor_trans(1).^i) .* weights_hist(end);
% end









