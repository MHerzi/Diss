% INPUT:
% - nc: Anzahl der Copulas in der Mixture Struktur
% - s1: Länge des Datensets
% - tv_faktor: Parameter des ARMA(1,1) Prozesses der zeitvariablen Gewichte
% - density: Dichten der Copulas aus der Mixture Struktur
% - funktion: entweder 'kalibrieren' oder 'vorhersage'
% - horizon: Zeithorizont der Vorhersage
%
% OUTPUT:
% - weights_tv: Gewichte der Mixture Copula im Zeitverlauf
% - PD: nx1 Vektor der Proportional Density
%
% HINWEIS:
% Funktion zur Bestimmung von zeitvariablen Mixture Copula Gewichten. Die
% Mixture Gewichte folgen einem ARMA(1,1) Prozess ohne Konstante. 
% Dabei werden immer AnzahlCopula-1 Gewichte übergeben, da sich
% das letzte Copulagewicht aus der Differenz 1-(Summe der anderen
% Copulagewichte) ergibt. Die Summe der Gewichte muss 1 ergeben. Der ARMA
% Prozess der Gewichte wird ähnlich Pattons (2006) Gleichung 1/(1-e^-x)
% invertiert um sicherzustellen dass die einzelnen Gewichte in den Grenzen
% [0,1] liegen. 
% Die AR-Faktoren beziehen sich auf die Mixture Gewichte in t-1. Die
% Ma-Faktoren beziehen sich auf den 10-Zeiteinheiten Durchschnitt der
% relativen Copuladichten.
%
%   Author: Valentin Braun, Martin Grizska
%   Phd Student in finance Goethe Universität, LMU
function [weights_tv, PD] = copulamix_Weights_tv_2(tv_faktor, nc, s1, density);

if nc > 1
    
    [d1, d2] = size(density);
    MALags = 10; % Zeitfenster über das der Durchschnitt der Gewichte gebildet wird, der mit dem ma-Faktor gewichtet wird
    weights_ARMA = zeros(s1-1,nc-1); % Platzhalter für untransformierte Gewichte aus dem ARMA Prozess
    weights_ARMA_trans = zeros(s1-1,nc-1); % Platzhalter für transformierte Gewichte aus dem ARMA Prozess; Patton (2006)
    weights_1 = tv_faktor(end-nc+1 : end)'; % Startwerte der Gewichte auslesen
    weights_ARMA(1,1:nc-1) = weights_1(1,1:nc-1);% -log(1./weights_1(1,1:nc-1) -1); % Startwerte der logarithmierten Gewichte übergeben
    k_W = tv_faktor(1); % Konstante des ARMA(1,1) Prozess 
    ar_W = tv_faktor(2:nc); % AR-Faktoren
    ma_W = tv_faktor(1+nc:1+2*(nc-1)); % MA-Faktoren
    
%     % ARMA Prozess wird auf Basis der absoluten Abstände der Indices berechnet
%     % ACHTUNG!!!! Hierfür müssen in copulaLL_mix_tv_2 die U[0 1] Daten anstelle
%     % der CopulaDichten eingelesen werden
%     for i = 2:s1
%         % Totale Differenz: k-Mean Algorithmus mit absoluten Differenzen:
%         % Die Daten werden in 1 Cluster pro Dimension eingeteilt und ihr jeweiler Abstand vom
%         % Clustermedian berechnet (K-Means Methode). Anschließend kann die
%         % Summe der absolute Summe der Abstände pro Cluster (AGD) berechnet werden.
%         % Dies ist letztendlich der gesamte Abstand. Um zu vermeiden dass die
%         % Dimensionsen den Zeitreihen entsprechen und folglich die Abstände pro
%         % Zeitreihe von ihrem Median berechnet würde, müssen die Dimensionen
%         % einen Zeitpunkt abbilden der die jeweiligen Datenpunte aller
%         % Zeitreihen behinhält. D.h. zum Zeitpunkt t wird der Median über die
%         % Datenpunkte der Zeitreihen gebildet. Daher muss die 1.Dim mit der
%         % 2.Dim der Zeitreihen vertauscht werden. Der Abstand jeder Zeitreihe zum
%         % Zeitpunkt t vom Median der Zeitreihen wird dann berechnet. Dies wird
%         % für jeden Zeitpunkt wiederholt und zum Schluss die absolute Summe der
%         % Abstände gebildet.
%         % Da hier die Gewichte berechnet werden, ohne eine Transformation der
%         % hier erzeugten Ergebnisse, wird die Summe der absoluten Abstände
%         % relativ zur Anzahl der Indices berechnet. D.h. der durchschnittl.
%         % Abstand pro Index vom Mittelpunkt des Cluster wird ausgegeben.
%         % Hierdurch liegt der Abstand immer im Bereich [0, 1], weil die
%         % Datengrundlage aus [0, 1] -Daten besteht.
%         if i <= MALags % Zugreifen auf kürzeren Zeitrahmen als MALags für K-Means Berechnung
%             [Cluster,Median,AGD(i),Distance] = kmeans(data(1:MALags,:)',1,'distance','cityblock');  
%             AGD(1) = AGD(2);
%         else
%             [Cluster,Median,AGD(i),Distance] = kmeans(data(i-MALags:i-1,:)',1,'distance','cityblock'); % 'sqEuclidean' nimmt den Mean aller Datenpunkte als Mittelpunkt
%         end
%         GD = AGD(i)./(MALags.*d2); % die AGD muss noch durch die LagAnzahl und die Indexanzhal dividiert werden.
%         % Berechnen des ARMA(1,1) Prozess
%         weights_ARMA(i,:) = k_W + ar_W .* weights_ARMA(i-1,:) + ma_W .* GD; 
%     end
    
    % ARMA Prozess wird auf Basis der relativen Copuladichten berechnet 
    % ACHTUNG!!!! Hierfür müssen in copulaLL_mix_tv_2 die CopulaDichten anstelle
    % der U[0 1] Daten eingelesen werden
    for i = 2:s1
        if i <= MALags
            PD(i) = mean(density(1:MALags-1, 1:nc-1), 1) ./ mean(sum(density(1:MALags-1, :), 2), 1); % Proportional Density
            weights_ARMA(i,:) = k_W + ar_W .* weights_ARMA(i-1,:) + ma_W .* PD(i); 
        else
            PD(i) = mean(density(i-MALags:i-1, 1:nc-1), 1) ./ mean(sum(density(i-MALags:i-1, :), 2), 1); 
            weights_ARMA(i,:) = k_W + ar_W .* weights_ARMA(i-1,:) + ma_W .* PD(i);           
        end
    end
    % weights_ARMA_trans = 1./(1+exp(-weights_ARMA)); % zeitvariable Gewichte werden nach Patton (2006) transformiert
    weights_tv = [weights_ARMA, 1-sum(weights_ARMA, 2)]; % Gewichte der Mixture Copula im Zeitverlauf
    
    PD = PD(:); % als nx1 Vektor ausgeben

else
    weights_tv = ones(s1,1); % Wenn nur 1 zeitvariable Copula kalibriert wird, wird keine Gewichtung vorgenommen
end
