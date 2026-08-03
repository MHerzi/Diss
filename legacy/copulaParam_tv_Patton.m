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
%   Author: Martin Grziska based on acode of Valentin Braun

function [CopParam_tv] = copulaParam_tv_Patton(family,tv_faktor,CopParam_1,data,funktion,horizon);

MALags = 10; % Patton (2006)

[s1 s2] = size(data);
family = lower(family);
tv_faktor = tv_faktor(:)';

% Vorhersage Horizont wird für MLE auf 0 gesetzt
if strcmp(funktion, 'kalibrieren')
	horizon = 0; 
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
if sum(strcmp(family, {'clayton' 'gumbel'})) ~= 1
    error('angegebene Copulafunktion kann nicht benutzt werden');
end

% Umrechnen des ersten Kappas in Kendalls Tau
CopParam_tv = zeros(s1+horizon, 1); % Platzhalter anlegen
CopParam_tv(1) = copulastat(family, CopParam_1); % Ersten CopulaParamter übergeben
% kappa(1) = copulastat(family, kappa(1));

% Erwartungswert der Residuen ist 0. Daher werden für horizon
% Zeiteinheiten 0 an die gefilterten Residuen angehängt
data = [data; zeros(horizon, s2)];

%--------------------------------------------------------------------------
% Clayton Copula
%--------------------------------------------------------------------------
% Berechnen der zeitabhängigen Kappas bedingt durch die Faktoren des ARMA
% Prozesses
for n = 2:s1+horizon
%     % Summe der absoluten Differenzenmatrix bilden:
%     % % Matrix der absoluten Differenzen der einzelnen Zeitreihen bilden (DM). Danach
%     % berechnen der absoluten DifferenzenMatrix (ADM). Hier werden die Differenzen aller
%     % Zeitreihen für die LagPerioden zueinander berechnet. Anschließend wird die Summe aller
%     % Differenzen gebildet. Dazu wird die Summe der oberen Dreicksmatrix
%     % gebildet. Abschließend wird die Summe durch die Anzahl der
%     % Zeitreihen geteilt um die Relativität der Differenzen zu
%     % wahren.
%     for i = 1:s2
%         if n<MALags
%             MS(i,:) = sum(abs(repmat(data(1:n,i),1,s2) - data(1:n,:)), 1);
%         else
%             MS(i,:) = sum(abs(repmat(data(n-MALags+1:n,i),1,s2) - data(n-MALags+1:n,:)), 1);
%         end
%     end
%     ADM = MS./MALags;
%     % Berechnen der Gesamtdifferenz (GD) aus der ADM
%     GD = sum(sum(triu(ADM, 1))); 
    
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
        [Cluster,median,AGD(n),Distance] = kmeans(data(1:n-1,:)',1,'distance','cityblock');  
    elseif n > s1+1 
        % Für Prognoserahmen wird die durchschnittl. Distanz d. letzten
        % MALags Zeitreihen benutzt um perfekte Abhängigkeit zu vermeiden.
        % Dies wäre der Fall wenn alle Erwartungswerte mit 0 modelliert
        % werden würden
        AGD(n) = mean(AGD(n-MALags:n-1));
    else
        [Cluster,median,AGD(n),Distance] = kmeans(data(n-MALags:n-1,:)',1,'distance','cityblock');
    end
    GD = AGD(n)./MALags; % die AGD muss noch durch die LagAnzahl dividiert werden. Patton (2006)
    
    % Patton (2006) erweitert auf den multidimensionalen Fall
    psi = tv_faktor(1) + tv_faktor(2)*CopParam_tv(n-1) + tv_faktor(3)*GD;
    CopParam_tv(n) = 1/(1+exp(-psi));		% Logistic Transformation nach Patton (2006)

% alternative Parametrisierung (Parameter wird nicht als Kendall's Tau
% interpretiert
% if strcmp(family,'clayton')
%     CopParam_tv(n) = 1/(1+exp(-psi));
% elseif strcmp(family,'gumbel')
%     CopParam_tv(n) = 1+1/(1+exp(-psi));
% end
end

% Output generieren
if strcmp(funktion, 'kalibrieren')
% 	CopParam_tv = kappa;  % gefilterten time-path des konditionalen Copulaparameters
elseif strcmp(funktion, 'vorhersage')
    CopParam_tv = CopParam_tv(s1+1:s1+horizon);  % prognostizierten time-path des konditionalen Copulaparameters
end
CopParam_tv = CopParam_tv(:);


