function [AR_pred, ht_pred, resid] = univariateForecastBacktest(daten,GARCHOutput_Backtest,start,forecastP,Spec)

[t,k]=size(daten);
if GARCHOutput_Backtest{1}{1}.const==1 %wenn ein GARCH-Modell mit Konstante geschätzt wurde, wurden es alle
    const=1;
end
for i=1:k
    ardiff(i)=GARCHOutput_Backtest{1}{i}.arlag-1;%über die verschiedenen samples sind die uniavriaten GARCH-Modelle gleich
end
% -------------------------------------------------------------------------
%                          univariate GARCH forecast
% ------------------------------------------------------------------------

g=1;
l=length(daten);
% in der letzen Zeile steht der letzte Forecast von t-forecastP auf t;
% nehme hier wieder den Startzeitpunkt für den vollen Datensatz - nicht uum
% die AR-Längen korrigiert
for i=start:forecastP:l-forecastP 
    AR_pred_all{g} = univariateAR_pred(daten(1:i+forecastP,:),GARCHOutput_Backtest{g},const); %zähle maxarlag hinzu, da ForecastStart sich auf die korrigierte Zeitreihe bezieht bei dem AR-Forecast wird aber noch einmal um die AR-laglänge korrigiert
    g=g+1;
end

g=1;
for i=start:forecastP:l-forecastP
    [ht_pred_all{g} resid_all{g}] = univariateHt_pred(daten(1:i+forecastP,:),GARCHOutput_Backtest{g},const,Spec.uniforecastP);
    g=g+1;
end

% Bringe die Daten der jeweiligen Forecast-Periode in eine Zeitreihe - im
% Beispiel steht also von 1:start_new + Länge des forecast (also z.B. 20 Perioden) die Daten, die mit dem ersten Satz
% Koeffizienten geschätzt wurden, danach folgen 20 Daten, die mit dem
% zweiten Satz Koeffizienten geschätzt wurden usw.; beachte die 20 Perioden
% sind 20 1-periodige forecasts
% füge die unterscheidlichen Zeiträume (mit den unterschiedlich geschätzten
% Parametern) in eine Zeitreihe (pro Datenzeitreihe) zusammen

numb = 1:forecastP;
for j=1:size(AR_pred_all,2)
    for h=1:k
        AR_pred(numb(1):numb(end),h) = AR_pred_all{j}{h}(end-forecastP+1:end)';
        ht_pred(numb(1):numb(end),h) = ht_pred_all{j}{h}(end-forecastP+1:end)';
        resid(numb(1):numb(end),h) = resid_all{j}{h}(end-forecastP+1:end)';
    end
    numb=numb+forecastP;
end
