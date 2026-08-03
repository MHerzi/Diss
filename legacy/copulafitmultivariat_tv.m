% Schätzen dynamischer multivariater Copulas
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - data: [0 1] verteilte Zeitreihen; max. 10 Zeitreihen
%   t: Zeitvariation modelliert durch ADCC
%
% OUTPUT:
% - CopParam_tv: CopulaParameter über den Zeitverlauf
% - tv_faktor: Parameter des ARMA Prozesses für den zeitvariablen
% CopulaParameter
% - LL: maximierter LogLikelihood Wert
% - StrOutput: exitflag, HesseMatrix
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function [CopParam_tv, tv_faktor, LL, StrOutput, Tstat, robustSE] = copulafitmultivariat_tv(family, data)

family = lower(family);
[s1 s2] = size(data);

% Prüfen dass Daten als ZeitxIndices eingeben werden
if s2>s1
    data = data';
    [s1 s2] = size(data);
end

% Prüfen dass Daten [0 1] verteilt sind
if max(max(data))>1 || min(min(data))<0
    error('Daten müssen [0 1] verteilt sein');
end

% Prüfen dass max. 10 Indices verwendet werden
if s2<2 || s2>10
    error('Es können minimal 2 und maximal 10 Zeitreihen in den dynamischen Copulas verwendet werden');
end

% Prüfen dass nur erlaubte Copulafamilien verwendet werden
if sum(strcmp(family, {'clayton' 'gumbel' 't' 'gaussian'})) ~= 1
    error('angegebene Copulafunktion kann nicht benutzt werden');
end

%--------------------------------------------------------------------------
% Auswählen der Copulafamilie und schätzen der zeitvariablen
% Copulaparameter
%--------------------------------------------------------------------------
switch family
    case {'gumbel' 'clayton'} % zeitvariabler Parameter wird entsprechend Patton (2006) kalibriert
        CopParam_1 = copulafitmultivariat(family, data); % Statische Copula definiert den ersten CopulaParameter im Zeitablauf
        % Transformieren des statischen CopulaParameters in zeitvariablen
        % Korrelationskoeffizienten als Konstante des ARMA Prozesses. ARMA
        % Variablen werden 0 gesetzt, damit die erste LL der zeitvariablen
        % Copula der optimierten LL der statischen Copula entspricht.
        x0 = [-log(1/copulastat(family, CopParam_1) -1);0;0]; 
        LB = -100*ones(3,1); % Limits werden nur gesetzt damit fmincon verwendet werden kann, ansonsten bedeutungslos
        UB = 100*ones(3,1);
        nonLinCon = []; % Funktion für nicht linearen Constraint
        P = []; % Lags der Residuen werden nur im ADCC Modell benötigt
        Q = []; % Lags der Covarianzmatrix werden nur im ADCC Modell benötigt
        A=[];
        b=[];
        
    case 't' % Skalarversion
        % Startwerte des DoF der t-Copula berechnen
        [x0_Corr x0_DoF] = copulafit(family, data);
        % xo-Aufteilung: Formel 5, Capiello et al.(2006): a,g,b, DoF der t-Copula
        P = 1; % Lags der Residuen
        Q = 1; % Lags der Covarianzmatrix
        x0 = [.1 .1 .8 x0_DoF]';
        UB = [0.99; 0.9; 0.999; 100];
        LB = [zeros(3,P)+2*10^-12; 2.1];
        CopParam_1 = [];
        nonLinCon = {'nonLinCon_Skalar_tv'}; % Funktion für nicht linearen Constraint
        A = [1 0 1 0; -1 -1 0 0];
        b = [1 0]-1e-5;
        
    case 'gaussian' % Skalarversion
        % xo-Aufteilung: Formel 5, Capiello et al.(2006): a,g,b, DoF der t-Copula
        P = 1; % Lags der Residuen
        Q = 1; % Lags der Covarianzmatrix
        x0 = [.1 .1 .8]';
        UB = [0.99; 0.9; 0.999];
        LB = zeros(3,P)+2*10^-10;
        CopParam_1 = [];
        nonLinCon = {'nonLinCon_Skalar_tv'}; % Funktion für nicht linearen Constraint
        A = [1 0 1; -1 -1 0];
        b = [1 0]-1e-5;
        

end

%--------------------------------------------------------------------------
% Optimierung 
%--------------------------------------------------------------------------
options = optimset('fmincon');
% options = optimset('Algorithm','interior-point','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
%     'TolCon',10^-12,'TolFun',10^-8,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);
options = optimset(options,'Algorithm','interior-point','Display','iter','MaxIter',500,'MaxFunEvals',10000,'Hessian','bfgs',...
    'TolCon',10^-12,'TolFun',10^-8,'TolX',10^-10, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);

% switch family
%     case {'gumbel' 'clayton'} % zeitvariabler Parameter wird entsprechend Patton (2006) kalibriert
%         [tv_faktor, likhood,exitflag,output,lambda,grad,hessian] =... 
%             fmincon(@(tv_faktor) copulaLL_tv(family, CopParam_1, tv_faktor, data), x0,A,b,[],[],LB,UB,[],options);
%         [LL, CopParam_tv] = copulaLL_tv(family, CopParam_1, tv_faktor, data);
%         
%     case {'t' 'gaussian'}
%         [tv_faktor, likhood,exitflag,output,lambda,grad,hessian] =... 
%             fmincon('copulaLL_tv',x0,[],[],[],[],LB,UB,nonLinCon,options, CopParam_1, family, data, P, Q);        
%         [LL, CopParam_tv] = copulaLL_tv(tv_faktor, CopParam_1, family, data, P, Q);
% end

[tv_faktor, likhood,exitflag,output,lambda,grad,hessian] =... 
fmincon('copulaLL_tv',x0,A,b,[],[],LB,UB,nonLinCon,options, CopParam_1, family, data, P, Q);        
[LL, CopParam_tv] = copulaLL_tv(tv_faktor, CopParam_1, family, data, P, Q);
[Tstat,robustSE] = TstatCopula(tv_faktor, CopParam_1, family, data, P, Q);
if exitflag>0
    display('erfolgreiche Optimierung der zeitvariablen Copula')
end
LL = -LL;

%--------------------------------------------------------------------------
% Output generieren
%--------------------------------------------------------------------------
StrOutput.exitflag = exitflag;
StrOutput.Hesse = hessian;

%--------------------------------------------------------------------------
% Graphische Darstellung der CopulaParameter im Zeitverlauf
%--------------------------------------------------------------------------
switch family
    case {'clayton' 'gumbel'}
        figure, hold 'on'
        plot(CopParam_tv, 'k')
        plot(repmat(CopParam_1,1,s1),'k --')
        legend('time-variable CopParam', 'time-stable CopParam', 'location','NorthWest');
         title([upper(family) '-Copula']);
    case {'t' 'gaussian'}
        for i = 1:s1
            Corr_tv(:,i) = nonzeros(triu(CopParam_tv(:,:,i)+10,1))-10; % Korrelationsparameter in Spaltenform transferieren
        end
        figure
        plot(Corr_tv')
        title(['time-variable Correlation Coefficients ' upper(family) '-Copula']);
end































