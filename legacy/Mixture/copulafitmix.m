% Schätzen dynamischer multivariater Mixture-Copulas
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - data: [0 1] verteilte Zeitreihen; max. 10 Zeitreihen
%
% OUTPUT:
% - CopParam_tv: CopulaParameter über den Zeitverlauf
% - tv_faktor: Parameter des ARMA Prozesses für den zeitvariablen
%   CopulaParameter
% - LL_Mix: maximierter LogLikelihood Wert der Mixture-Copula
% - LL: LogLikelihood Wert der einzelnen Copulas in der Mixture Struktur
% - StrOutput: exitflag, HesseMatrix
% - AIC: Akaike Information Kriterium
% - BIC: Bayesian Information Kriterium
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
function [CopParam_tv, Weights_tv, tv_faktor, LL_Mix, LL, AIC, BIC] = copulafitmix(family, data)

% Prüfen der Input Parameter
[nc, family, s1, s2] = copulamix_tv_parameter_check(family, data);

%--------------------------------------------------------------------------
% Auswählen der Copulafamilien und festlegen der Startparameter sowie der
% Grenzwerte
%--------------------------------------------------------------------------
for i = 1:nc
    switch family{i}
        case {'gumbel' 'clayton'} % zeitvariabler Parameter wird entsprechend Patton (2006) kalibriert
            CopParam_1(i) = copulafitmultivariat(family{i}, data); % Statische Copula definiert den ersten CopulaParameter im Zeitablauf
            if nc == 1
                % Transformieren des statischen CopulaParameters in zeitvariablen
                % Korrelationskoeffizienten als Konstante des ARMA Prozesses. ARMA
                % Variablen werden 0 gesetzt, damit die erste LL der zeitvariablen
                % Copula der optimierten LL der statischen Copula entspricht.
                x0{i} = [-log(1/copulastat(family{i}, CopParam_1(i)) -1);0;0]'; 
            elseif nc > 1
                [m1, m2, ArchCopParam_tv] = copulafitmix_tv(family{i}, data);
                clear m1 m2
                x0{i} = ArchCopParam_tv';
            end
            LB{i} = -100*ones(1,3); % Limits werden nur gesetzt damit fmincon keine komplexen Zahlen erzeugt, ansonsten bedeutungslos
            UB{i} = 100*ones(1,3);
            nonLinCon{i} = []; % Funktion für nicht linearen Constraint
            P{i} = []; % Lags der Residuen werden nur im ADCC Modell benötigt
            Q{i} = []; % Lags der Covarianzmatrix werden nur im ADCC Modell benötigt
            weights_1(i) = 0.1; % Startgewichte

        case 't' % Skalarversion
            % Startwerte des DoF der t-Copula berechnen
            [x0_Corr x0_DoF] = copulafit(family{i}, data);
            % xo-Aufteilung: Formel 5, Cappiello et al.(2006): a,g,b, DoF der t-Copula
            P{i} = 1; % Lags der Residuen
            Q{i} = 1; % Lags der Covarianzmatrix
            x0{i} = [0.1 0.1 0.8 x0_DoF];
            UB{i} = [0.9; 0.9; 0.99; 200]';
            LB{i} = [1e-3; 1e-6; .5; 2.1]';
            nonLinCon{i} = {'nonLinCon_Skalar_mix_tv'}; % Funktion für nicht linearen Constraint
            CopParam_1(i) = 0; % Platzhalter wird nur angegeben wenn keine der Copulas in der Mixture Struktur archimedisch ist
            weights_1(i) = 0.9; % Startgewichte
            
        case 'gaussian' % Skalarversion
            % xo-Aufteilung: Formel 5, Capiello et al.(2006): a,g,b
            P{i} = 1; % Lags der Residuen
            Q{i} = 1; % Lags der Covarianzmatrix
            x0{i} = [0.1 0.1 0.8];
            UB{i} = [0.9; 0.9; 0.99]';
            LB{i} = [1e-6; 1e-6; .5]';
            nonLinCon{i} = {'nonLinCon_Skalar_mix_tv'}; % Funktion für nicht lineare Constraint
            CopParam_1(i) = 0; % Platzhalter wird nur angegeben wenn keine der Copulas in der Mixture Struktur archimedisch istCopParam_1 = [];
            weights_1(i) = 0.9; % Startgewichte

    end
end

% Startwerte der Gewichte und Grenzwerte erzeugen
weights_1 = weights_1(:)./sum(weights_1); % Startgewichte
% weights_1 = ones(nc,1)./nc; % Startgewichte
LB_W = -ones(1,2*(nc-1)) .* 10; % Untergrenze der zeitvariablen GewichtsParameter, damit fmincon keine Extremwerte konstruiert und dort hängenbleibt
UB_W = ones(1,2*(nc-1)) .* 10; % Obergrenze der zeitvariablen GewichtsParameter, damit fmincon keine Extremwerte konstruiert und dort hängenbleibt
if nc>1
    x0_W = zeros(1,2*(nc-1)); % Startwerte der zeitvariablen Parameter
else
    x0_W = [];
end

% Zusammenfügen der Startwerte und Grenzwerte aller zeitvariablen StrukturParameter der
% Copulas aus der Mixture Struktur und hinzufügen der ARMA(ar,ma) Parameter
% der Mixture Gewichte. Daher kommen zu den Startwerten noch 2*(nc-1)
% Parameter für die zeitvariablen Gewichte hinzu
LB_total = [LB{1:end} LB_W]'; 
UB_total = [UB{1:end} UB_W]';
x0_total = [x0{1:end} x0_W]';
nonLinCon_total = [nonLinCon{1:end}]; % Wird nur für einstufige Schätzung benötigt
% CopParam_1_total = [CopParam_1{1:end}]; % Wird nur für einstufige Schätzung benötigt
P_total = [P{1:end}]; % Wird nur für einstufige Schätzung benötigt
Q_total = [Q{1:end}]; % Wird nur für einstufige Schätzung benötigt
    
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
[tv_faktor, likhood,exitflag,output,lambda,grad,hessian] =... 
    fmincon('copulaLL_mix_tv',x0_total,[],[],[],[],LB_total,UB_total,nonLinCon_total,options, CopParam_1, weights_1, family, data, P_total, Q_total);        
[LL_Mix, LL, CopParam_tv, Weights_tv, density, likelihoods_mix, likelihoods] = copulaLL_mix_tv(tv_faktor, CopParam_1, weights_1, family, data, P_total, Q_total);

% % zweistufige Optimierung: zeitvariable CopulaParameter jeder Copula werden getrennt
% % ermittelt. Anschließend werden die Parameter des ARMA Prozesses der zeitvariablen Gewichte geschätzt.
% for i = 1:nc
%     [tv_faktor{i}, LL_Einzel,exitflag] =... 
%         fmincon('copulaLL_mix_tv',x0{i},[],[],[],[],LB{i},UB{i},nonLinCon{i},options, CopParam_1(i), 1, {family{i}}, data, P{i}, Q{i});        
%     [LL_Einzel, LL, CopParam_tv, weights_Einzel_tv, density(:,i)] = copulaLL_mix_tv(x0{i}, CopParam_1(i), 1, {family{i}}, data, P{i}, Q{i});
%     if exitflag>0
%         display(['erfolgreiche Optimierung der zeitvariablen ', upper(family{i}), '-Copula'])
%     end
% end
% % Schätzen der zeitvariablen GewichtsParameter
% if nc>1
%     [Weights_tv, LL_Weights, exitflag] = fmincon('copulamix_LL_Weights_tv',x0_W,[],[],[],[],LB_W,UB_W,[],options, nc, s1, weights_1, density);        
%     if exitflag>0
%             display(['erfolgreiche Optimierung der zeitvariablen GewichtsParameter der Mixture-Copula'])
%     end
% else
%     Weights_tv = [];
% end
% % Berechnen LL der Mixture Copula
% tv_faktor_total = [tv_faktor{1:end} Weights_tv];
% [LL_Mix, LL, CopParam_tv, Weights_tv] = copulaLL_mix_tv(tv_faktor_total, CopParam_1, weights_1, family, data, P_total, Q_total);

%--------------------------------------------------------------------------
% Output generieren
%--------------------------------------------------------------------------
if exitflag>0
    display('erfolgreiche Optimierung der zeitvariablen Copula')
end
LL = -LL;   
LL_Mix = -LL_Mix;

% Akaike und BIC Kriterium berechnen
np = size(tv_faktor(:), 1); % Anzahl der Mixture Parameter auslesen
[AIC, BIC] = aicbic(LL_Mix, np, s1*s2);

% StrOutput.exitflag = exitflag;
% StrOutput.Hesse = hessian;

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
                Corr_tv(:,i) = nonzeros(triu(CopParam_tv{j}(:,:,i)+10,1))-10; % Korrelationsparameter in Spaltenform transferieren
            end
            figure % zeitvariable Korrelationskoeffizienten in Matrixform graphisch darstellen
            n11 = 0; n22 = 0;
            for n1 = 1:s2-1
                for n2 = 1:s2-n1
                    n11 = n11+1;
                    n22 = n22+1;
                    subplot(s2-1,s2-1,n22)
                    plot(Corr_tv(n11,:), 'k')
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

