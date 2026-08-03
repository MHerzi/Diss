function [mu,h,resid_ar]=Univariate_ar_mean(daten,nlags)
% Berechnet die optimale AR-Länge und gibt den mit der optimalen AR-Länge
% berechneten mu einer Zeitreihe wieder
% Input:
%        daten: (Return-)Zeitreihe
%        nlags: maximale Länge der AR-order 
% Output: mu (der Wert für die jeweilige Zeitreihe steht in der jeweiligen
%         h gibt die optimale Laglänge an
% Spalte)

[t,k]=size(daten);

for i=1:k
    for j=1:nlags
        resultsaic(i).j = ar_modified(daten(:,i),j);
        aic_lag1(i,j)=t*log(resultsaic(i).j.sumsquare/(t-(j+1)))+2*(j+1);
    end
end

for i=1:k
    for j=1:nlags
        lagy=mlag(daten(:,i),j);
        lagy=lagy(j+1:size(lagy,1),1);
        datenlag=daten(j+1:size(daten,1),i);
        resultsaiclag(i).j=ols_modified(datenlag,lagy);
        aic_lag2(i,j)=t*log(resultsaiclag(i).j.sumsquare/(t-(j+1)))+2*(j+1);
    end
end

aic=[aic_lag1; aic_lag2];

[m,h]=min(aic');

% Schätze AR mit optimale Lag-Länge
for i=1:k
    result(i)=ar_modified(daten(:,i),h(i));
    mu(i)=result(i).beta(1)./(1-sum(result(i).beta(2:end)));
    size_resid(:,i)=size(result(i).resid,1);
end

min_resid=min(min(size_resid));

for i=1:k
    if size(result(i).resid,1)>min_resid
        result(i).resid = result(i).resid(size(result(i).resid,1)-min_resid+1:size(result(i).resid,1),1);
        resid_ar(:,i) = result(i).resid;
    else resid_ar(:,i) = result(i).resid;
    end
end