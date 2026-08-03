% INPUT:
% - nc: Anzahl der Copulas in der Mixture Struktur
% - s1: Länge des Datensets
% - weights_1: Startwerte der Gewichte
% - tv_faktor: Parameter des ARMA(1,1) Prozesses der zeitvariablen Gewichte
% - density: Dichten der Copulas aus der Mixture Struktur
% - funktion: entweder 'kalibrieren' oder 'vorhersage'
% - horizon: Zeithorizont der Vorhersage
%
% OUTPUT:
% - weights_tv: Gewichte der Mixture Copula im Zeitverlauf
% - MD: letzte Beobachtung Mean Density
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
function [weights_tv, MD] = copulamix_Weights_tv_grm(tv_faktor, nc, s1, weights_1, density, Dynamic);

if nc > 1
    MALags = 10; % Zeitfenster über das der Durchschnitt der Gewichte gebildet wird, der mit dem ma-Faktor gewichtet wird
    weights_ARMA = zeros(s1-1,nc-1); % Platzhalter für untransformierte Gewichte aus dem ARMA Prozess
    weights_ARMA_trans = zeros(s1-1,nc-1); % Platzhalter für transformierte Gewichte aus dem ARMA Prozess; Patton (2006)
    weights_ARMA_trans(1,1:nc-1) = weights_1(1,1:nc-1); % Startwerte der Gewichte übergeben
    ar_W = tv_faktor(1:(nc-1)); % AR-Faktoren
    ma_W = tv_faktor(1+(nc-1):2*(nc-1)); % MA-Faktoren
    
    % Gewichte im Zeitverlauf via ARMA Prozess berechnen
    for i = 2:s1
        if i <= MALags
            MD = mean(density(1:i-1, 1:nc-1), 1) ./ mean(sum(density(1:i-1, :), 2), 1); % Mean Density
            weights_ARMA(i,:) = ar_W .* weights_ARMA_trans(i-1,:) + ma_W .* MD; 
        else
            MD = mean(density(i-MALags:i-1, 1:nc-1), 1) ./ mean(sum(density(i-MALags:i-1, :), 2), 1); 
            weights_ARMA(i,:) = ar_W .* weights_ARMA_trans(i-1,:) + ma_W .* MD;
        end
        weights_ARMA_trans(i,:) = 1./(1+exp(-weights_ARMA(i,:))); % zeitvariable Gewichte werden nach Patton (2006) transformiert
    end
    weights_tv = [weights_ARMA_trans, 1-sum(weights_ARMA_trans, 2)]; % Gewichte der Mixture Copula im Zeitverlauf
else
    weights_tv = ones(s1,1); % Wenn nur 1 zeitvariable Copula kalibriert wird, wird keine Gewichtung vorgenommen
end
