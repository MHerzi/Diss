% Schätzen dynamischer multivariater Mixture-Copulas
% Die dynamischen Prozesse (ARMA(1,1)) der archimedischen Copulas werden
% ARMA konform restringiert. |AR|<1 und |MA|<1. Gleiches gilt auch für die
% Parameter des Gewichte ARMA Prozess. Die zugehörigen Constraints werden
% so gesetzt dass KEINE Transformation der Gewichte mehr nötig ist.
% Hierdurch wird die multistep Prognose weiter erleichtert.
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - Cop_Stat: 'an' oder 'aus' indexiert ob die Standardfehler und t-Werte
% berechnet werden
% - data: [0 1] verteilte Zeitreihen; max. 10 Zeitreihen
%
% OUTPUT:
% - CopParam_tv: CopulaParameter über den Zeitverlauf
% - tv_faktor: Parameter des ARMA Prozesses für den zeitvariablen
% CopulaParameter
% - LL_Mix: maximierter LogLikelihood Wert der Mixture-Copula
% - LL: LogLikelihood Wert der einzelnen Copulas in der Mixture Struktur
% - StrOutput: exitflag, HesseMatrix
% - AIC: Akaike Information Kriterium
% - BIC: Bayesian Information Kriterium
% - Tstat: t-Statistik
% - robustSE: robuste Standardfehler nach White (1982)
%
% Hinweis:
% die zeitvariablen Modellparameter sind für alle Copulafamilien und
% Modelle auf Skalare beschränkt. Bei der Gauss und t-Copula werden auch
% nur Skalare verwendet und nicht die Diagonalmatrizen oder vollständig
% flexiblen Matrizen wie in Capiello et al. (2006). Dies wäre zu
% rechenintensiv!
% Die Gewichte der zeitvariablen Mixture Copula folgen einem ARMA Prozess
%
% Anwendung:
% family: wird übergeben als Cell-Struktur
% Es können maximal 2 CopulaFamilien verwendet werden und minimal 1
% CopulaFamilie. 1 CopulaFamilie entspricht demselben Prozess wie
% copulafitmultivariat_tv
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function [CopParam_tv, Weights_tv, tv_faktor, LL_Mix, LL, AIC, BIC, SE_robust p_Wert t_Wert] = copulafitmix_tv_2(family, Cop_Stat, data);

% Prüfen der Input Parameter
[nc, family, s1, s2] = copulamix_tv_parameter_check(family, data);

% Festlegen dass max. 2 Copulas implementiert werden
if nc > 2
    error('max. 2 Copulas möglich')
end

%--------------------------------------------------------------------------
% Auswählen der Copulafamilien und festlegen der Startparameter sowie der
% Grenzwerte
%--------------------------------------------------------------------------
for i = 1:nc
    switch family{i}
        % zeitvariabler Parameter wird ähnlich zu Patton (2006) kalibriert,
        % jedoch mit zusätzlichen Bedingungen die die Stationarität und
        % Invertierbarkeit des ARMA Prozesses garantieren          
        case {'gumbel' 'clayton'} 
            if nc == 1
                CopParam_1(i) = copulafitmultivariat_grm(family{i}, data); % Statische Copula definiert den ersten CopulaParameter im Zeitablauf
                if isnan(CopParam_1(i)) && s2==2 %f�r bivariate Daten (DVine mixture) funktioniert die obige Funktion manchmal nicht, benutze dann Matlab
                    CopParam_1(i) = copulafit(family{i},data);
                end
                % Transformieren des statischen CopulaParameters in zeitvariablen
                % Korrelationskoeffizienten als Konstante des ARMA Prozesses. ARMA
                % Variablen werden 0 gesetzt, damit die erste LL der zeitvariablen
                % Copula der optimierten LL der statischen Copula entspricht.
                if strcmp(family{i}, 'gumbel')
                    x0{i} = [log(CopParam_1(i) - 1);0;0]'; 
                elseif strcmp(family{i}, 'clayton')
                    x0{i} = [log(CopParam_1(i));0;0]'; 
                end
                nonLinCon{i} = {'nonLinCon_ArchCop_tv_2'}; % Wird nur für einzelne Kalibrierung der archimedischen Copula benötigt
            elseif nc > 1
                [m1, m2, ArchCopParam_tv] = copulafitmix_tv_2(family{i}, 'aus', data);
                CopParam_1(i) = m1{1}(1); % Startparameter der archimedischen Copula übergeben
                clear m1 m2
                x0{i} = ArchCopParam_tv(1:3)'; % Startwerte für ARMA Prozess
            end
            % |AR|<1 & |MA|<0; Bedingung für Stationarität und
            % Invertierbarkeit; Nötig für multistep Prognosen
            LB{i} = [-10, -0.9999, -0.9999]; 
            UB{i} = [10, 0.9999, 0.9999];
%             nonLinCon{i} = []; % Funktion für nicht linearen Constraint
            P{i} = []; % Lags der Residuen werden nur im ADCC Modell benötigt
            Q{i} = []; % Lags der Covarianzmatrix werden nur im ADCC Modell benötigt
            weights_1(i) = 0.1; % Startgewichte
            if strcmp(family{i}, 'gumbel')
                Density_Func{i} = copulapdffunc(family{i}, s2); % Gumbel Dichtefunktion herleiten und als Funktion abspeichern. Steigert Zeiteffizienz 
            elseif strcmp(family{i}, 'clayton')
                Density_Func{i} = []; % Leere DichteFunktion übergeben
            end

        case 't' % Skalarversion
            % Startwerte des DoF der t-Copula berechnen
            [x0_Corr x0_DoF] = copulafit(family{i}, data);
            % xo-Aufteilung: Formel 5, Capiello et al.(2006): a,g,b, DoF der t-Copula
            P{i} = 1; % Lags der Residuen
            Q{i} = 1; % Lags der Covarianzmatrix
            x0{i} = [0.05 0.01 0.9 x0_DoF]; % Die Summe der Startwerte sollte < 1 sein
            UB{i} = [0.99; 0.99; 0.99; 200]';
            LB{i} = [1e-3; 0; 0.8; 2.1]';
            nonLinCon{i} = {'nonLinCon_Skalar_mix_tv_2'}; % Funktion für nicht linearen Constraint
            CopParam_1(i) = 0; % Platzhalter wird nur angegeben wenn keine der Copulas in der Mixture Struktur archimedisch ist
            weights_1(i) = 0.9; % Startgewichte
            Density_Func{i} = []; % Leere DichteFunktion übergeben
            
        case 'gaussian' % Skalarversion
            % xo-Aufteilung: Formel 5, Capiello et al.(2006): a,g,b
            P{i} = 1; % Lags der Residuen
            Q{i} = 1; % Lags der Covarianzmatrix
            x0{i} = [0.05 0.01 0.9]; % Die Summe der Startwerte sollte < 1 sein
            UB{i} = [0.99; 0.99; 0.99]';
            LB{i} = [1e-3; 0; 0.8]';
            nonLinCon{i} = {'nonLinCon_Skalar_mix_tv_2'}; % Funktion für nicht lineare Constraint
            CopParam_1(i) = 0; % Platzhalter wird nur angegeben wenn keine der Copulas in der Mixture Struktur archimedisch istCopParam_1 = [];
            weights_1(i) = 0.9; % Startgewichte
            Density_Func{i} = []; % Leere DichteFunktion übergeben

    end
end

% Zusammenfügen Grenzwerte aller zeitvariablen StrukturParameter der
% Copulas aus der Mixture Struktur und hinzufügen der ARMA(ar,ma) Parameter
% der Mixture Gewichte. Daher kommen zu den Startwerten noch 2*(nc-1)
% Parameter für die zeitvariablen Gewichte hinzu
% Ober- und Untergrenze der zeitvariablen GewichtsParameter werden so gewählt dass
% der ARMA(1,1) Prozess stationär und invertierbar ist. |AR|<1 & |MA|<1
% Zuletzt werden noch Beschränkungen für die Startgewichte festgelegt.
% Constraints für ARMA(1,1) Gewichte Prozess
% 1. k<= |ar|+|ma| (nonlinear)
% 2. 0 <= k <= 1 wird in lin constraints abgedeckt
% 3. -0.9999 <= ma, ar <= 0.9999
% 4. k+ar+ma <= 1
% 5. k+ar <= 1
% 6. k+ma <= 1
if nc > 1 % Mixture Copula
    LB_W = [0, -ones(1,2*(nc-1)) .* 0.9999, zeros(1,nc)]; 
    UB_W = [0.98, ones(1,2*(nc-1)) .* 0.9999, ones(1,nc)]; 
%     LB_W = [-10, -ones(1,2*(nc-1)) .* 0.9999, zeros(1,nc)]; 
%     UB_W = [10, ones(1,2*(nc-1)) .* 0.9999, ones(1,nc)]; 
elseif nc == 1
    LB_W = [];
    UB_W = [];
end
P_total = [P{1:end}]; % Lags der Residuen zusammenführen
Q_total = [Q{1:end}]; % Lags der Covarianzmatrix zusammenführen
LB_total = [LB{1:end} LB_W]'; 
UB_total = [UB{1:end} UB_W]';
nonLinCon_total = [nonLinCon{1:end}]; % Übergeben der nicht linearen Constraints als Funktion
% Platzhalter für lineare Constraints anlegen. Diese werden nur bei Mixture
% Copulas benötigt.
LC_eq = [];
lc_eq = [];
LC_noneq = [];
lc_noneq = [];

% Startwerte der Gewichte via Simulationen erzeugen
if nc>1 % Mixture Copulas
    % Lineare Constraints für Startgewichte
    LC_eq = zeros(size(LB_total));
    LC_eq = LC_eq(:)'; % Sicherstellen dass Equality Constraints als 1xn Vektor vorliegen
    LC_eq(end-nc+1:end) = 1; % lineare Constraints der Startgewichte definieren
    lc_eq = 1; % Sicherstellen dass die Summe der Startgwichte genau 1 ergibt
    LC_noneq = zeros(3, size(LB_total));
    LC_noneq(1, end-nc-2:end-nc) = 1; % Sicherstellen dass K+AR+MA <= 1
    LC_noneq(2, end-nc-2:end-nc-1) = 1; % Sicherstellen dass K+AR+MA <= 1
    LC_noneq(3, [end-nc-2, end-nc]) = 1; % Sicherstellen dass K+AR+MA <= 1
    lc_noneq = [1;1;1]; % k+ar+ma <= 1
    % Simulieren der Startparameter
    fprintf(1,'\n  Startparameter für dynamische Gewichte simulieren ... \n');
    n_RND = 100; % Anzahl der Zufallssimulationen
    zufall = rand(n_RND, 2*(nc-1)); % Zufallszahlen für Gewichte erzeugen
    weights_rnd = zufall./repmat(sum(zufall, 2), 1, 2*(nc-1)); % zufällige Startgewichte berechnen
    % Zufallswerte für die Parameter des dynamischen Prozess simulieren. 
    % AR & MA Parameter bleiben immer zwischen [-0.9999, 0.9999].
    % Konstante des ARMA(1,1) Prozess kann zwischen [-10, 10] pendeln
%     x0_W = [-10 + 20.*rand(n_RND,1), -0.9999 + 1.9999.*rand(n_RND,2*(nc-1))]; % Simulation der ARMA(1,1) Parameter
    x0_W = [0.7, 0.1, 0.1]; % Simulation der ARMA(1,1) Parameter
    for i = 1:n_RND
%         x0_total_Pre(:,i) = [x0{1:end} x0_W(i,:) weights_rnd(i,:)]'; % Startgwichte werden mit optimiert, daher in Startvektor übergeben
        x0_total_Pre(:,i) = [x0{1:end} x0_W weights_rnd(i,:)]'; % Startgwichte werden mit optimiert, daher in Startvektor übergeben
        LL_Mix_Pre(i) = copulaLL_mix_tv_2(x0_total_Pre(:,i), CopParam_1, family, data, P_total, Q_total, Density_Func);
    end 
    x0_total = x0_total_Pre(:, LL_Mix_Pre==min(LL_Mix_Pre)); % Startparameter
else % Einzelne Copulas
    x0_total = [x0{1:end}]';
end


%--------------------------------------------------------------------------
% Optimierung 
%--------------------------------------------------------------------------
% Optimierungsparameter festlegen
options = optimset('fmincon');
% options = optimset(options, 'Algorithm','interior-point','Display','iter','MaxIter',500,'MaxFunEvals',10000,'Hessian','bfgs',...
%     'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);
options = optimset(options, 'Algorithm','trust-region-reflective','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
    'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);

% einstufige Optimierung: zeitvariable CopulaParameter aller Copulas und Mixture
% Gewichte werden in einer Schätzung gemeinsam ermittelt.
[tv_faktor,likhood,exitflag,output,lambda,grad,hessian] =... 
    fmincon('copulaLL_mix_tv_2',x0_total,LC_noneq,lc_noneq,LC_eq,lc_eq,LB_total,UB_total,nonLinCon_total,options, CopParam_1, family, data, P_total, Q_total, Density_Func);        
[LL_Mix, LL, CopParam_tv, Weights_tv, density] = copulaLL_mix_tv_2(tv_faktor, CopParam_1, family, data, P_total, Q_total, Density_Func);

%--------------------------------------------------------------------------
% Output generieren
%--------------------------------------------------------------------------
if exitflag>0
    display('----- erfolgreiche Optimierung der dynamischen Copula -----');
end
LL = -LL;   
LL_Mix = -LL_Mix;

% Akaike und BIC Kriterium berechnen
np = size(tv_faktor(:), 1); % Anzahl der Mixture Parameter auslesen
[AIC, BIC] = aicbic(LL_Mix, np, s1*s2);

% StrOutput.exitflag = exitflag;
% StrOutput.Hesse = hessian;

%--------------------------------------------------------------------------
% Berechnen der Standardfehler der Copula Parameter
%--------------------------------------------------------------------------
% Standard Fehler werden in 1xn Vektor passend zu param Vektor gespeichert
fprintf(1,'\n  Standard Fehler berechnen ... \n');
[Cov_Mat SE_robust p_Wert t_Wert] = copulaStat_mix_tv_2(Cop_Stat, tv_faktor, CopParam_1, weights_1, family, data, P_total, Q_total);

%--------------------------------------------------------------------------
% Graphische Darstellung der CopulaParameter im Zeitverlauf
%--------------------------------------------------------------------------
for j = 1:nc
    switch family{j}
        case {'clayton' 'gumbel'}
            figure, hold 'on'
            plot(CopParam_tv{j}, 'k -')
            plot(repmat(CopParam_1(j),1,s1),'k --')
            legend('time-variable CopParam', 'time-stable CopParam', 'location','NorthWest');
            if nc == 1
                title(['Single ' upper(family{j}) ' Copula']);
            elseif nc > 1
                title(['Mixture ' upper(family{j}) ' Copula']);
            end
        case {'t' 'gaussian'}
%             set(gcf,'DefaultAxesColorOrder',[0 0 0],'DefaultAxesLineStyleOrder','-|--|-.|:'); % Festlegen der Farbe und Linientyps
            for i = 1:s1
                Corr_tv(:,i) = nonzeros(triu(CopParam_tv{j}(:,:,i)+10,1)')-10; % Korrelationsparameter in Spaltenform transferieren
            end
            figure % zeitvariable Korrelationskoeffizienten in Matrixform graphisch darstellen
            n11 = 0; n22 = 0;
            for n1 = 1:s2-1
                for n2 = 1:s2-n1
                    n11 = n11+1;
                    n22 = n22+1;
                    subplot(s2-1,s2-1,n22)
                    plot(Corr_tv(n11,:), 'k')
                    axis tight
                    axis 'auto y'
%                     title(['time-variable Correlation Coefficients ' upper(family{j}) '-Copula']);
                end
                n22 = n22+s2-n2;
            end    
            % Graphische Darstellung aller Korrelationskoeffizienten im
            % Zeitveraluf in einer Graphik
            figure
            plot(Corr_tv')
            title(['time-variable Correlation Coefficients ' upper(family{j}) ' Copula']);
    end
end
if nc>1
    figure
    set(gcf,'DefaultAxesColorOrder',[0 0 0],'DefaultAxesLineStyleOrder','-|--|-.|:'); % Festlegen der Farbe und Linientyps
    plot(Weights_tv, 'MarkerSize',3); % Gewichte im Zeitverlauf
end

























