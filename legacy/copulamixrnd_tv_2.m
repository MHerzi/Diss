% Version 2:
% Funktion zum Simulieren aus der dynamischen Mixture Copula. 
% Diese Funktion bezieht sich auf copulamix_tv_paramforecast_2, da hier die
% ARMA Parameter der dynamischen Archimedischen Copulas und des Gewicht
% Prozess dem Stationaritäts- und Invertierbarkeitskriterium folgen. Daher
% können hiermit multiperioden Forecasts generiert werden.
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - tv_faktor: Parameter des ARMA Prozesses des Copulaparameters
% - data: [0 1] verteilte Zeitreihen
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% - n_Sim: Anzahl der Simulationen pro Spalte
% - horizon: Zeithorizont der Simulationen. Bsp.: horizon=10 -> die
% CopulaParameter der dynamischen Mixture Copula werden für den Zeitpunt
% t+10 prognostiziert. Mit diesen CopulaParametern werden n_Sim
% Simulationen aus der dynamischen mixture Copula generiert
% P,Q werden nur für die elliptischen Copulas übergeben!
%
% OUTPUT:
% - sim: aus der dynamischen Mixture Copula simulierte gleichverteilte Daten
% - CopParam_pred: CopulaParameter zum Zeitpunkt [t+1,...,t+horizon]
% - weights_pred: prognostizierte Gewichte
%
%   Author: Valentin Braun 
%   Phd Student in finance Goethe Universität
function [sim, CopParam_pred, Weights_pred] = copulamixrnd_tv_2(tv_faktor,family,data,n_Sim,horizon);

% Prüfen der Input Parameter
[nc, family, s1, s2] = copulamix_tv_parameter_check(family, data);
tv_faktor = tv_faktor(:);

% Lags der Kovarianzmatrix; Kann später auch auf mehrere Dimension erhöht werden, 
% aber muss mit Kalibrierung abgestimmt sein!
P = 1; 
Q = 1;

% -----------------------------------------------------------------------
% Vorhersage der CopulaParameter zum Zeitpunkt (t+1) 
% -----------------------------------------------------------------------
[CopParam_pred, Weights_pred] = copulamix_tv_paramforecast_2(tv_faktor, nc, family, data, P, Q, horizon);

% -----------------------------------------------------------------------
% Simulieren aus der statischen Mixture Copula
% -----------------------------------------------------------------------
% Nachdem die CopulaParameter für die Zeitpunkte (t+1,...,t+n) im letzten Schritt
% berechnet wurden können diese nun in den Simulationsalgorithmus der
% statischen Mixture Copula übergeben werden. Dadurch wird für jeden
% Zeitpunkte (t+1,...,t+n) ein Datenset bestehend aus jeweils n_Sim [0 1]
% verteilten Simulationen generiert. Die CopulaParameter werden zum
% jeweiligen Zeitpunkt entsprechend aus den Vorhersagen ausgelsen und in
% den statischen Simulationsalgorithmus übergeben. Dadurch wird zu jedem
% Zeitpunt aus einer "anderen" Copula simuliert, was dem dynamischen
% Gedanken entspricht.
for i = 1:horizon
    clear Coeff
    for j = 1:nc
        switch family{j}
            case 'gaussian'
                Coeff{j} = CopParam_pred{j}(:,:,i); % Korrelationsmatrix übergeben
            case 't'
                Coeff{j}{1} = CopParam_pred{j}{1}(:,:,i); % Korrelationsmatrix übergeben
                Coeff{j}{2} = CopParam_pred{j}{2}; % statischen Freiheitsgrad übergeben
            case {'gumbel' 'clayton'}
                Coeff{j} = CopParam_pred{j}(i); % archimedischen CopulaParameter übergeben
        end
    end
    sim(:,:,i) = copulamixrnd(family, Coeff, Weights_pred(i,:), s2, n_Sim); % Simulation aus der statischen Mixture Copula mit den prognostizierten CopulaParametern
end















