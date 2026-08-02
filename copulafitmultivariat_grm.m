% Anpassen multivariater Copulas. Zulässige Copula Familien sind: 
% t, Gauss, Frank, Clayton, Gumbel
%
%   Author: Valentin Braun, Martin Grziska
%   Phd Student in finance Goethe Universität
function [Output LL exitflag numParam] = copulafitmultivariat_grm(family, data, CopStat);

family = lower(family);
[s1 s2] = size(data);

% Prüfen dass zumindest ein bivariates Datenset vorliegt
if s2<2
    error('Datenset muss zumindest bivariat sein um Copulas verwenden zu können');
end

% Prüfen dass data nur uniforme Daten beinhaltet
if max(max(data))>1 || min(min(data))<0
    error('Daten~[0 1] ist nicht gegeben');
end

% Prüfen dass nur zulässige Copulas verwendet werden
if not(strcmp(family, {'t' 'gaussian' 'clayton' 'gumbel' 'frank' 'rotclayton'}))
    error('die eingegebene Copula kann nicht verwendet werden');
end

% Prüfen dass bei archimedischen Copulas max 19 Zeitreihen im Datenset sind
if strcmp(family, {'clayton' 'gumbel' 'frank' 'rotclayton'})
    if s2>19
        error('archimedische Copulas sind auf 14 Zeitreihen begrenzt');
    end
end

% Festlegen der Optimierungs Parameter
options=optimset('fmincon');
options = optimset(options,'Algorithm','interior-point','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
    'TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);

switch family
    
    case {'clayton' 'frank' 'gumbel' 'rotclayton'} % archimedische Copulas
        % Constraints für Optimierung
        switch family
            case {'clayton', 'rotclayton', 'frank'}
                lb = 1e-3;
            case 'gumbel'
                lb = 1;
        end
        ub = 50; %restringiere Copula-Abh�ngigkeitsparameter auf 50 da sonst numerische Probleme auftreten k�nne (50 stellt schon extreme Abh�ngigkeit da)
        Coeff0 = 1; % Startparameter wird für alle archimedischen Copulas auf 1 setzten
        % Symbolische Dichtefunktionen für archimedische  Copulas berechnen
        CopulaDensityFunc = copulapdffunc_grm(family, s2);
        % Minimierung der LogLikelihood Funktion 
        display('copulafitmultivariat LL Minimierung ...');
        if s2>2 
            [CopParam LL exitflag]=fmincon(@(Coeff) copulaarchLL(CopulaDensityFunc, data, Coeff),Coeff0,[],[],[],[],lb,ub,[],options);
            if exitflag>0
                display('----- erfolgreiche Optimierung der Copula Parameter -----');
                LL=-LL;
            end
        else % bei bivariaten Daten nutzen Matlab Copula-funktion 
            if strcmp(family,'clayton') || strcmp(family,'rotclayton')
                CopParam=copulafit('clayton',data);
            elseif strcmp(family,'gumbel')
                CopParam=copulafit('gumbel',data);
            end
        end
        Output = CopParam;
        % Anzahl der CopulaParameter auslesen
        numParam = numel(CopParam);
        
    case {'t' 'gaussian'}
        % Es wird nicht die Korrelationsmatrix in die Optimierung
        % übergeben. Stattdessen werden die Cholesky Parameter benutzt um
        % sicherstellen zu können dass der MLE Algorithmus immer pos.
        % definite Matrizen produziert.
        Rho = corrcoef(data);
        Coeff0 = nonzeros(chol(Rho)')'; % Cholesky Faktoren in Vektorform übergeben
        % Grenzwerte festlegen
        lb = -0.99.*ones(1, (s2^2-s2)/2+s2);
        ub = 0.99.*ones(1, (s2^2-s2)/2+s2);
        if strcmp(family, 't')
            Coeff0 = [Coeff0 1]; % Bei der t-Copula muss noch ein Freiheitsgrad mit in die Optimierung übergeben werden
            lb = [lb 2.1]; % unterer Grenzwert für Freiheitsgrad
            ub = [ub 200]; % oberer Grenzwert für Freiheitsgrad
        end
        
        % Für die non-linear Constraints müssen einige Parameter global übergeben werden
        warning('off');
        global s2_global family_global;
        s2_global = s2;
        family_global = family;
        
        % Minimierung der LogLikelihood Funktion 
        display('copulafitmultivariat LL Minimierung ...');
        [CopParam LL exitflag]=fmincon(@(Coeff) copulaellipticLL(family, data, Coeff),Coeff0,[],[],[],[],lb,ub,@nlcmultivariat,options);
        if exitflag>0
            display('----- erfolgreiche Optimierung der Copula Parameter -----');
            LL=-LL;
        end
        % Anzahl der CopulaParameter auslesen
        numParam = numel(CopParam);
        % Cholesky Koeffizienten der Gauss und t-Copula in Korrelationsmatrizen
        % umrechnen. Die Copula Parameter werden in der Output Struktur ausgegeben
        switch family
            case 't'
                Output{1} = chol2corr(s2, CopParam(1:end-1)); % Cholesky Faktoren in Korrelationsmatrix umrechnen und übergeben
                Output{1} = validcorr(Output{1}); % Sicherstellen dass dies eine tatsächliche Korrelationsmatrix ist.
                Output{2} = CopParam(end); % Freiheitsgrad der t-Copula
            case 'gaussian'
                Output = chol2corr(s2, CopParam); % Cholesky Faktoren in Korrelationsmatrix umrechnen und übergeben
                Output = validcorr(Output); % Sicherstellen dass dies eine tatsächliche Korrelationsmatrix ist.
        end
end
        
end

%--------------------------------------------------------------------------
% archimedische Copula LL 
%--------------------------------------------------------------------------
function LL = copulaarchLL(CopulaDensityFunc, data, Coeff);
[s1 s2] = size(data);
% Splitten der Daten in einzelne Vektoren v1...vN
clear v v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 % Max. Variablen = 10
v = [];
for i = 1:s2
   var = genvarname('v', who);
   eval([var ' = data(:, i);'])
end
% Berechnen der Dichten für archimedische Copulas berechnen
switch s2
    case 1
        error('Min # Indices = 2');
    case 2
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2);
    case 3
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3);
    case 4
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4);
    case 5
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5);
    case 6
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6);
    case 7
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7);
    case 8
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8);
    case 9
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9);
    case 10
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10);
    case 11
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11);
    case 12
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);
    case 13
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13);
    case 14
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14);
    case 15
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15);
    case 16
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16);
    case 17
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17);
    case 18
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v18);
    case 19
        CopulaDensity = CopulaDensityFunc(Coeff, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v19);
        
        
        
        
end
clear v v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19
% negative LL berechnen
LL = -sum(log(CopulaDensity), 1);
end

%--------------------------------------------------------------------------
% elliptische Copula LL 
%--------------------------------------------------------------------------
function LL = copulaellipticLL(family, data, Coeff);
[s1 s2] = size(data);
% Korrelationsmatrix aus Coeff auslesen
switch family
    case 't'
        CF = Coeff(1:end-1); % Cholesky Faktoren auslesen. Der letzte Wert in der Coeff Struktur ist der Freiheitsgrad bei der t-Copula
        Rho = chol2corr(s2, CF); % Berechnen der Korrelationsmatrix via Cholesky Faktoren 
        DoF = Coeff(end); % Freiheitsgrad auslesen
        CopulaDensity = copulapdfmultivariat_grm(family, data, Rho, DoF); 
    case 'gaussian'
        CF = Coeff; % Cholesky Faktoren auslesen. 
        Rho = chol2corr(s2, CF); % Berechnen der Korrelationsmatrix via Cholesky Faktoren 
        CopulaDensity = copulapdfmultivariat_grm(family, data, Rho);
end
% neg. LL berechnen
LL = -sum(log(CopulaDensity), 1);
end

%--------------------------------------------------------------------------
% Non-Linear Constraints
%--------------------------------------------------------------------------
function [c, ceq] = nlcmultivariat(Coeff);
% Die Funktion stellt sicher dass die Korrelationsmatrizen in der Gauss und
% t-Copula immer pos. definit definiert sind. Hierzu wird das Verfahren
% nach Pelletier (2006) angewandt.
global s2_global; % Holen der Datenset Breite 
s2 = s2_global; 
global family_global; % Holen der Copulas in der Mixture Struktur
family = family_global;
% Input Parameter werden aus Inputvektor für fmincon ausgelesen 
% und in eine passende Struktur transferiert
switch family
    case 't'
        CF = Coeff(1:end-1); % Cholesky Faktoren übergeben. Freiheitsgrad wird nicht mit übernommen
    case 'gaussian'
        CF = Coeff; % Cholesky Faktoren übergeben. 
end
% Prüfen dass die implizierten Cholesky Faktoren ein pos. definite Matrizen
% erzeugen.
c = 0; ceq = 0;
switch family
    case {'t' 'gaussian'}
        % Cholesky Faktoren aus Vektorenform in Matrix konvertieren und
        % prüfen dass die zugehörig Korrelationsmatrix positiv definit
        % ist (PD_Test = 0 wenn dies der Fall ist)
        [KM, CFM, PD_Test] = chol2corr(s2, CF); % KM: Korrelationsmatrix, CFM: Cholesky Faktoren Matrix, PD_Test: Test auf positiv Definitheit
        ceq = ceq+PD_Test; % Nur wenn dies ein Nullvektor ist, sind alle Korrelationsmatrizen positiv definit.
end

end



