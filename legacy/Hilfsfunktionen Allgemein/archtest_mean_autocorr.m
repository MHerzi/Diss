function [result,autoLag,mu,H]=archtest_mean_autocorr(daten,nlags)
% Berechnet die Autokorrelationsfunktion und führt danach eine
% OLS-Schätzung durch
% Output:
% "result": Variable mit den Ergebnissen der OLS-Schätzung
% "autoLag": Autokorrelationsstruktur der Zeitreihe
% "mu": geschätzter Mittelwert der Zeitreihe
% "H": Variable, die Null oder Eins. Ljung-Box Q-Test, ob die Residuen der
% OLS-Schätzung white noise sind. Nullhypothese Residuen sind white noise
% (H=0). Konfidenzniveau: alpha=0.05
[t,k]=size(daten);

if k>1
    error('als Input darf nur ein Zeilenvektor verwendet werden!');
end
    
[ACF, Lags, Bounds] = autocorr(daten(:,1), nlags);

ACF=ACF(2:end,:);
m=1;
for i=1:nlags
    if abs(ACF(i,:))>=Bounds(1,1);
        autoLag(m)=i;
        m=m+1;
    end
end
    
for i=1:size(autoLag,2)
    xlag{i}=lagmatrix(daten,autoLag(i));
end

maxx=max(autoLag);
for i=1:size(xlag,2)
    xregress(:,i)=xlag{i}(maxx+1:end,1);
end
xregress=[ones(size(xregress,1),1) xregress];


yregress=daten(max(autoLag)+1:end,1);

result=ols(yregress,xregress);
mu=result.beta(1)/(1-sum(result.beta(2:end)));

[H, pValue, Qstat, CriticalValue] = lbqtest(result.resid,nlags);

if H~=0
    error('Residuen sind kein white noise')
end

% Check auf Stationarität
if abs(sum(result.beta(2:end))) >=1
    error('Zeitreihe ist nicht stationär')
end