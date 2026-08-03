% Funktion zur Berechnung des zeitvariablen Copulaparameters für
% archimedische Copulas. Dabei wird der Copulaparameter ähnlich
% eines ARMA(1,1) Prozesses modelliert; Patton(2006)
% Die Funktion erweitert Pattons Modell für den multidimensionalen Fall
% implementierte Copulas: Clayton, Gumbel
%
% INPUT:
% - tv_faktor: Parameter des ARMA Prozesses des Copulaparameters: erster
% Faktor: Konstante; zweiter Faktor: Lag des Copulaparameters; dritter
% Faktor: MA Prozess mit 10 Lags
% - CopParam_1: erster CopulaParameter im Zeitablauf. Sollten den Parametern
% der statischen Copula entsprechen
% - data: [0 1] verteiltes Datenset
% - family: Clayton, Gumbel
% - funktion: entweder 'kalibrieren' oder 'vorhersage'
% - horizon: Zeithorizont der Vorhersage
%
% OUTPUT:
% - CopParam_tv: zeitvariabler CopulaParamter
%
%   Author: Martin Grziska based on a code of Valentin Braun

function [CopParam_tv,GD] = copulaParam_tv_Patton_grm2(family,tv_faktor,CopParam_1,data,funktion,horizon);

MALags = 10; % Patton (2006)

[s1 s2] = size(data);
family = lower(family);
K = tv_faktor(1); % Konstante
AR = tv_faktor(2); % AR-Term
MA = tv_faktor(3); % MA-Term



% Vorhersage Horizont wird für MLE auf 0 gesetzt
if strcmp(funktion, 'kalibrieren')
	horizon = 0; 
end

% Prüfen dass Daten [0 1] verteilt sind
if max(max(data))>1 || min(min(data))<0
    error('Daten müssen [0 1] verteilt sein');
end

% Prüfen dass max. 10 Indices verwendet werden
if s2<2 || s2>14
    error('Es können minimal 2 und maximal 10 Zeitreihen in den dynamischen Copulas verwendet werden');
end

% Prüfen dass nur erlaubte Copulafamilien verwendet werden
if sum(strcmp(family, {'clayton' 'gumbel' 'rotclayton'})) ~= 1
    error('angegebene Copulafunktion kann nicht benutzt werden');
end

% Die Konstante des ARMA Prozess ist der transponierte Mean des
% Copulaparameters
psi(1) = tv_faktor(1);

CopParam_tv = zeros(s1+horizon, 1); % Platzhalter anlegen
% CopParam_tv(1) = CopParam_1;% Ersten CopulaParamter als Archimedischen Copulaparameter übergeben

% Erwartungswert der Residuen ist 0. Daher werden für horizon
% Zeiteinheiten 0 an die gefilterten Residuen angehängt
data = [data; zeros(horizon, s2)];

%--------------------------------------------------------------------------
% Archimedische Copulas
%--------------------------------------------------------------------------
% Berechnen der zeitabhängigen Kappas bedingt durch die Faktoren des ARMA
% Prozesses
for n = 2:s1+horizon
    % Totale Differenz: k-Mean Algorithmus mit absoluten Differenzen:
    % Die Daten werden in 1 Cluster pro Dimension eingeteilt und ihr jeweiler Abstand vom
    % Clustermedian berechnet (K-Means Methode). Anschließend kann die
    % Summe der absolute Summe der Abstände pro Cluster (AGD) berechnet werden.
    % Dies ist letztendlich der gesamte Abstand. Um zu vermeiden dass die
    % Dimensionsen den Zeitreihen entsprechen und folglich die Abstände pro
    % Zeitreihe von ihrem Median berechnet würde, müssen die Dimensionen
    % einen Zeitpunkt abbilden der die jeweiligen Datenpunte aller
    % Zeitreihen behinhält. D.h. zum Zeitpunkt t wird der Median über die
    % Datenpunkte der Zeitreihen gebildet. Daher muss die 1.Dim mit der
    % 2.Dim der Zeitreihen vertauscht werden. Der Abstand jeder Zeitreihe zum
    % Zeitpunkt t vom Median der Zeitreihen wird dann berechnet. Dies wird
    % für jeden Zeitpunkt wiederholt und zum Schluss die absolute Summe der
    % Abstände gebildet.
%     sum(sum(abs(X'-repmat(median(X'),4,1))))
    if n <= MALags % Zugreifen auf kürzeren Zeitrahmen als MALags für K-Means Berechnung
        [Cluster,Median,AGD(n),Distance] = kmeans(data(1:MALags,:)',1,'distance','cityblock');  
        AGD(1) = AGD(2);
    elseif n > s1+1 
        % Für Prognoserahmen wird die durchschnittl. Distanz d. letzten
        % MALags Zeitreihen benutzt um perfekte Abhängigkeit zu vermeiden.
        % Dies wäre der Fall wenn alle Erwartungswerte mit 0 modelliert
        % werden würden
        AGD(n) = mean(AGD(n-MALags:n-1));
    else
        [Cluster,Median,AGD(n),Distance] = kmeans(data(n-MALags:n-1,:)',1,'distance','cityblock'); % 'sqEuclidean' nimmt den Mean aller Datenpunkte als Mittelpunkt
    end
    GD(n) = AGD(n)./MALags; % die AGD muss noch durch die LagAnzahl dividiert werden. Patton (2006)
    
    % Patton (2006) erweitert auf den multidimensionalen Fall via GD. 
    psi(n) = K + AR*psi(n-1) + MA*GD(n);
end

% Die logistische Transformation wird in VERÄNDERTER Form für die
% archimedischen Copulas berechnet 
switch family
    case {'clayton', 'rotclayton'}
        CopParam_tv = exp(psi) + 1e-6; % Sicherstellen dass der Parameter > 0
    case 'gumbel'
        CopParam_tv = exp(psi) + 1 + 1e-6; % Sicherstellen dass der Parameter > 1
end




GD = GD(:); % sicherstellen dass ein nx1 Vektor pbergeben wird

% Output generieren
if strcmp(funktion, 'kalibrieren')
% 	CopParam_tv = kappa;  % gefilterten time-path des konditionalen Copulaparameters
elseif strcmp(funktion, 'vorhersage')
    CopParam_tv = CopParam_tv(s1+1:s1+horizon);  % prognostizierten time-path des konditionalen Copulaparameters
end

% % sicherstellen, dass Copula-parameter <= 20 ist (der Copula-Parameter steht
% % als Exponent in der Clayton und Gumbel-Copula. Ist der Exponent zu gro�,
% % so kann Matlab teilweise keine numerischen Berechnungen durchf�hren
% CopParam_tv(CopParam_tv>20)=20;

CopParam_tv = CopParam_tv(:);


