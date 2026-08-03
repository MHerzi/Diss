% Funktion zur einstufigen Schätzung der Mixture Gewichte
%
% INPUT:
% - nc: Anzahl der Copulas in der Mixture Struktur
% - s1: Länge des Datensets
% - weights_1: Startwerte der Gewichte
% - tv_faktor: Parameter des ARMA(1,1) Prozesses der zeitvariablen Gewichte
% - density: Dichten der Copulas aus der Mixture Struktur
%
% OUTPUT:
% - LL_Mix: Likelihood Wert der Mixture Copula
%
%   Author: Valentin Braun, Martin Grizska
%   Phd Student in finance Goethe Universität, LMU
function [LL_Mix] = copulamix_LL_Weights_tv(tv_faktor, nc, s1, weights_1, density);

% Falls Mixture Struktur vorliegt, folgen die Gewichte einem ARMA(1,1) Prozess. 
% Berechnen der zeitvariablen Gewichte
 [weights_tv] = copulamix_Weights_tv(tv_faktor, nc, s1, weights_1, density);

% LL für die Mixture Copula berechnen. Hierzu werden die Dichten der
% einzelnen Copulas gewichtet, anschließend logarithmiert und aufaddiert
MixDensity = sum(weights_tv .* density, 2); 
LL_Mix = -sum(log(MixDensity));
% Sicherstellen dass LL_Mix nicht NaN oder komplex wird. Dies würde einen
% Fehler in fmincon verursachen
if isnan(LL_Mix) || ~isreal(LL_Mix)
        LL_Mix = 1e6;
end