% Schätzen dynamischer multivariater Mixture-Copulas
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - data: [0 1] verteilte Zeitreihen; max. 10 Zeitreihen
% - Cop_Stat: 'an' oder 'aus' indexiert ob die Standardfehler und t-Werte
% berechnet werden
% (stderrros: string - 'on' wenn berechnet werden sollen, sonst [])
%
% OUTPUT:
% - CopParam_tv: CopulaParameter über den Zeitverlauf
% - tv_faktor: Parameter des ARMA Prozesses für den zeitvariablen
%        CopulaParameter (!!! DCC-Parameter d�rfen nicht mehr quadriert werden,
%        alle anderen Parameter der verschiedenen Modelle m�ssen zur
%        Vergleichbarkeit quadriert werden!)
% - LL_Mix: maximierter LogLikelihood Wert der Mixture-Copula
% - LL: LogLikelihood Wert der einzelnen Copulas in der Mixture Struktur
% - StrOutput: exitflag, HesseMatrix
% - AIC: Akaike Information Kriterium
% - BIC: Bayesian Information Kriterium
% - SE: Strukturvariable mit Standardfehlern
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
%   Author: Martin Grziska based on a code of Valentin Braun
% �nderung: die kmeans Spezifikation wurde durch die Originalspezifikation
% Patton ersetzt
function [CopParam_tv, Weights_tv, tv_faktor, LL_Mix, LL, AIC, BIC, CopParam_1, SE, x0] = copulafitmix_tv_grm2_var(family, data, Dynamic, Cop_Stat, PrtFig)


% Prüfen der Input Parameter
[nc, family, s1, s2] = copulamix_tv_parameter_check_grm(family, data);


%--------------------------------------------------------------------------
% Auswählen der Copulafamilien und festlegen der Startparameter sowie der
% Grenzwerte
%--------------------------------------------------------------------------
for i = 1:nc
    switch family{i}
        case {'gumbel' 'clayton' 'rotclayton'} % zeitvariabler Parameter wird entsprechend Patton (2006) kalibriert
            if strcmp(family{i},'rotclayton')
                dataarchi = 1-data; %wenn rotated Clayton verwendet wird m�ssen unif(0,1)-Daten gedreht werden
            else
                dataarchi = data;
            end
            CopParam_1(i) = copulafitmultivariat_grm(family{i}, dataarchi); % Statische Copula definiert den ersten CopulaParameter im Zeitablauf
            if nc == 1
                % Transformieren des statischen CopulaParameters in zeitvariablen
                % Korrelationskoeffizienten als Konstante des ARMA Prozesses. ARMA
                % Variablen werden 0 gesetzt, damit die erste LL der zeitvariablen
                % Copula der optimierten LL der statischen Copula
                % entspricht.
                if strcmp(family{i}, 'gumbel')
                    x0{i} = [log(CopParam_1(i) - 1);0;0]';
                elseif strcmp(family{i}, 'clayton') || strcmp(family{i}, 'rotclayton')
                    x0{i} = [log(CopParam_1(i));0;0]';
                end
                nonLinCon{i} = {'nonLinCon_ArchCop_tv_2'}; % Wird nur für einzelne Kalibrierung der archimedischen Copula benötigt
            elseif nc > 1
                [m1, m2, ArchCopParam_tv] = copulafitmix_tv_grm2(family{i}, data, Dynamic, 'aus', 'off');
                if s2>2
                    CopParam_1(i) = m1{1}(1); % Startparameter der archimedischen Copula übergeben
                elseif s2==2
                    CopParam_1(i) = copulafit(family{i},data);
                end
                clear m1 m2
                x0{i} = ArchCopParam_tv(1:3)'; % Startwerte für ARMA Prozess
            end
            % |AR|<1 & |MA|<0; Bedingung für Stationarität und
            % Invertierbarkeit; Nötig für multistep Prognosen
            %             original-Spezifikation
            LB{i} = [-100, -0.9999, -0.9999]; %neagtive Werte f�r Konstante k�nnen zugelassen werden, da Copula-Parameter noch exponentiell transformiert wird
            UB{i} = [100, 0.9999, 0.9999];
            nonLinCon{i} = []; % Funktion für nicht linearen Constraint
            P{i} = []; % Lags der Residuen werden nur im ADCC Modell benötigt
            Q{i} = []; % Lags der Covarianzmatrix werden nur im ADCC Modell benötigt
            weights_1(i) = 0.5; % Startgewichte
            %             A=[];
            %             b=[];
            if strcmp(family{i}, 'gumbel')
                Density_Func{i} = copulapdffunc_grm(family{i}, s2); % Gumbel Dichtefunktion herleiten und als Funktion abspeichern. Steigert Zeiteffizienz
            elseif strcmp(family{i}, 'clayton') || strcmp(family{i}, 'rotclayton')
                Density_Func{i} = []; % Leere DichteFunktion übergeben
            end

        case 't' % Skalarversion
            % Startwerte des DoF der t-Copula berechnen
            [x0_Corr x0_DoF] = copulafit(family{i}, data);
            % xo-Aufteilung: Formel 5, Capiello et al.(2006): a,g,b, DoF der t-Copula
            P{i} = 1; % Lags der Residuen
            Q{i} = 1; % Lags der Covarianzmatrix
            if strcmp(Dynamic,'ADCC')
                trdata=norminv(data);
                x0{i} = Est_ADCC(trdata,1,1);
                if x0{i}(end)<0.8 %wenn der GARCH-Parameter < 0.8 ist setze manuelle Werte
                    x0{i}=[];
                    x0{i} = [.1 .01 .8]';
                end
                x0{i}=[x0{i}' x0_DoF];
                UB{i} = [0.9; 0.9; 0.99; 200]';
                LB{i} = [1e-6; 1e-6; 1e-6; 2.1]';
            elseif strcmp(Dynamic,'DCC')
                 trdata=norminv(data);
                x0{i}= Est_DCC_cop_start(trdata,1,1);
                if x0{i}(end)<0.8 %wenn der GARCH-Parameter < 0.8 ist setze manuelle Werte
                    x0{i}=[];
                    x0{i} = [.1 .8]';
                end
                x0{i}=[x0{i}' x0_DoF];
                UB{i} = [0.9; 0.99; 200]';
                LB{i} = [1e-6; 1e-6; 2.1]';
            elseif strcmp(Dynamic,'GDCC')
                  trdata=norminv(data);
                x0{i} = Est_GDCC_cop_start(trdata,1,1);
                if sum(x0{i}(end-s2+1:end))<s2*0.8 %wenn die Summe der GARCH-Parameter < s2*0.8 ist nehme manuelle Werte
                    x0{i}=[];
                    x0{i} = [ones(1,s2)*0.1 ones(1,s2)*0.8]';
                end
                x0{i} = [x0{i}' x0_DoF];
                UB{i} = [ones(1,s2*2)*0.999 200];
                LB{i} = [zeros(1,s2*2)+1e-6 2.1];
            elseif strcmp(Dynamic,'AGDCC')
                trdata=norminv(data);
                x0{i} = Est_AGDCC_cop_start(trdata,1,1);
                if sum(x0{i}(end-s2+1:end))<s2*0.8 %wenn die Summe der GARCH-Parameter < s2*0.8 ist nehme manuelle Werte
                    clear x0
                    x0{i} = [ones(1,s2)*0.1 ones(1,s2)*0.01 ones(1,s2)*0.8]';
                end
                x0{i} = [x0{i}' x0_DoF];
                UB{i} = [ones(1,s2*3)*0.999 200];
                LB{i} = [zeros(1,s2*3)+1e-6 2.1];
            end
            nonLinCon{i} = {'nonLinCon_Skalar_mix_tv_grm2'}; % Funktion für nicht linearen Constraint
            CopParam_1(i) = 0; % Platzhalter wird nur angegeben wenn keine der Copulas in der Mixture Struktur archimedisch ist
            weights_1(i) = 0.5; % Startgewichte
            Density_Func{i} = []; % Leere DichteFunktion übergeben
            %             A = [1 1 zeros(1,6)]; % Constraint f�r DCC-Modell: a+b<1; Matlab: A*x0_total<b
            %             b = 1-1e-6;% Constraint f�r DCC-Modell: a+b<1; Matlab: A*x0_total<b

        case 'gaussian' % Skalarversion
            % xo-Aufteilung: Formel 5, Capiello et al.(2006): a,g,b
            P{i} = 1; % Lags der Residuen
            Q{i} = 1; % Lags der Covarianzmatrix
            if strcmp(Dynamic,'ADCC')
                trdata=norminv(data);
                x0{i} = Est_ADCC(trdata,1,1);
                if x0{i}(end)<0.8 %wenn der GARCH-Parameter < 0.8 ist setze manuelle Werte
                    x0{i}=[];
                    x0{i} = [.1 .01 .8]';
                end
                x0{i}=x0{i}';
                UB{i} = [0.9; 0.9; 0.99]';
                LB{i} = [1e-6; 1e-6; 1e-6]';
                nonLinCon{i} = {'nonLinCon_Skalar_mix_tv_grm2'}; % Funktion für nicht lineare Constraint
                CopParam_1(i) = 0; % Platzhalter wird nur angegeben wenn keine der Copulas in der Mixture Struktur archimedisch istCopParam_1 = [];
                weights_1(i) = 0.5; % Startgewichte
            elseif strcmp(Dynamic,'DCC')
                trdata=norminv(data);
                x0{i}= Est_DCC(trdata,1,1);
                if x0{i}(end)<0.8 %wenn der GARCH-Parameter < 0.8 ist setze manuelle Werte
                    x0{i}=[];
                    x0{i} = [.1 .8]';
                end
                x0{i}=x0{i}';
                UB{i} = [0.9; 0.999]';
                LB{i} = [1e-6; 1e-6]';
                nonLinCon{i} = {'nonLinCon_Skalar_mix_tv_grm2'}; % Funktion für nicht linearen Constraint
            elseif strcmp(Dynamic,'GDCC')
                trdata=norminv(data);
                x0{i} = Est_GDCC(trdata,1,1);
                if sum(x0{i}(end-s2+1:end))<s2*0.8 %wenn die Summe der GARCH-Parameter < s2*0.8 ist nehme manuelle Werte
                    x0{i}=[];
                    x0{i} = [ones(1,s2)*0.1 ones(1,s2)*0.8]';
                end
                x0{i} = x0{i}';
                UB{i} = ones(1,s2*2)-1e-6;
                LB{i} = zeros(1,s2*2)+1e-6;
                nonLinCon{i} = {'nonLinCon_Skalar_mix_tv_grm2'}; % Funktion für nicht lineare Constraint
            elseif strcmp(Dynamic,'AGDCC')
                trdata=norminv(data);
                x0{i} = Est_AGDCC(trdata,1,1);
                if sum(x0{i}(end-s2+1:end))<s2*0.8 %wenn die Summe der GARCH-Parameter < s2*0.8 ist nehme manuelle Werte
                    clear x0
                    x0{i} = [ones(1,s2)*0.1 ones(1,s2)*0.01 ones(1,s2)*0.8]';
                end
                x0{i} = x0{i}';
                UB{i} = ones(1,s2*3)-1e-6;
                LB{i} = zeros(1,s2*3)+1e-6;
                nonLinCon{i} = {'nonLinCon_Skalar_mix_tv_grm2'}; % Funktion für nicht lineare Constraint
            end
            CopParam_1(i) = 0; % Platzhalter wird nur angegeben wenn keine der Copulas in der Mixture Struktur archimedisch istCopParam_1 = [];
            weights_1(i) = 0.5; % Startgewichte
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
    LC_noneq = zeros(3, size(LB_total,1));
    LC_noneq(1, end-nc-2:end-nc) = 1; % Sicherstellen dass K+AR+MA <=1 (tv-Gewichte)
    LC_noneq(2, end-nc-2:end-nc) = -1; % Sicherstellen dass K+AR+MA >=0 (tv-Gewichte)
    LC_noneq(3, end-nc-2:end-nc-1) = 1; % Sicherstellen dass K+AR <= 1 (tv-Gewichte)
    LC_noneq(4, [end-nc-2, end-nc]) = 1; % Sicherstellen dass K+MA <= 1 (tv-Gewichte)
    lc_noneq = [1;0;1;1]; % k+ar+ma <= 1
    % Simulieren der Startparameter
    fprintf(1,'\n  Startparameter für dynamische Gewichte simulieren ... \n');
    if s1<=700
        n_RND = 100; % Anzahl der Zufallssimulationen
    else
        n_RND = 50;
    end
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
        LL_Mix_Pre(i) = copulaLL_mix_tv_grm2(x0_total_Pre(:,i), CopParam_1, family, data, P_total, Q_total, Dynamic, Density_Func);
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
options = optimset(options, 'Algorithm','trust-region-reflective','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
    'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);
% options = optimset(options, 'Algorithm','trust-region-reflective','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
%     'TolCon',10^-20,'TolFun',10^-20,'TolX',20^-10, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.3);

% einstufige Optimierung: zeitvariable CopulaParameter aller Copulas und Mixture
% Gewichte werden in einer Schätzung gemeinsam ermittelt.
epsilon = 10^(-5);
[tv_faktor,likhood,exitflag,output,lambda,grad,hessian] =...
    fmincon('copulaLL_mix_tv_grm2_var',x0_total,LC_noneq,lc_noneq,LC_eq,lc_eq,LB_total,UB_total,nonLinCon_total,options, CopParam_1, family, data, P_total, Q_total, Dynamic, Density_Func, epsilon);
if exitflag < 0
    options=optimset(options,'Algorithm','interior-point');
    [tv_faktor,likhood,exitflag,output,lambda,grad,hessian] =...
        fmincon('copulaLL_mix_tv_grm2',x0_total,LC_noneq,lc_noneq,LC_eq,lc_eq,LB_total,UB_total,nonLinCon_total,options, CopParam_1, family, data, P_total, Q_total, Dynamic, Density_Func, epsilon);
end
% if exitflag < 0
%     count = 1;
%     constraint1 = 10^-6;
%     constraint2 = 10^-2;
%     while exitflag < 0 && count <=5 %wenn keine korrekte L�sung gefunden wird ver�ndere die Optionen
%         clear options
%         options = optimset('fmincon');
%         options = optimset(options,'Algorithm','interior-point');
%         options = optimset(options,'TolCon',constraint1);
%         options = optimset(options,'TolX',constraint1);
%         options = optimset(options,'TolFun',constraint2);
%         [tv_faktor,likhood,exitflag,output,lambda,grad,hessian] =...
%             fmincon('copulaLL_mix_tv_grm2',x0_total,LC_noneq,lc_noneq,LC_eq,lc_eq,LB_total,UB_total,nonLinCon_total,options, CopParam_1, family, data, P_total, Q_total, Dynamic, Density_Func, epsilon);
%         constraint1 = constraint1/2;
%         constraint2 = constraint2/2;
%         epsilon = epsilon/2;
%         count = count+1;
%     end
% end

[LL_Mix, LL, CopParam_tv, Weights_tv, density] = copulaLL_mix_tv_grm2_var(tv_faktor, CopParam_1, family, data, P_total, Q_total, Dynamic, Density_Func, epsilon);

% durch numerische Probleme k�nnen Gewichte <0 sein; sie m�ssen aber >= 0
% sein

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
% Berechnen der Standardfehler der Copula Parameter
%--------------------------------------------------------------------------
% Standard Fehler werden in 1xn Vektor passend zu param Vektor gespeichert
fprintf(1,'\n  Standard Fehler berechnen ... \n');
[Cov_Mat SE_robust p_Wert t_Wert] = copulaStat_mix_tv_grm2(Cop_Stat, tv_faktor, CopParam_1, weights_1, family, data, P_total, Q_total, Dynamic);
SE.CovMat=Cov_Mat;
SE.SErobust = SE_robust;
SE.pWert = p_Wert;
SE.tWert = t_Wert;
%--------------------------------------------------------------------------
% Graphische Darstellung der CopulaParameter im Zeitverlauf
%--------------------------------------------------------------------------
if strcmp(PrtFig,'on')
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
end

