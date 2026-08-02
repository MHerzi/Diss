% % % % %
% Bestimme die Marginalverteilung für den HFRXEMN

clear all
% % Lese den HFRXEMN ein
cd G:\Daten
y = xlsread('Hedge_HFRX_20090611',1);
x=price2ret(y(:,2));

[t,k]=size(x);

% Skewness
skew=skewness(x);

% kurtosis
kurt=kurtosis(x);

% Checke auf Autokorrelation im Mean

results_x_mean = lmtest1(x,5); %H_0: es liegt keien Autokorrelation vor
% Ergebnis: Es liegt Autokorrelation im Mean vor

% % % Checke die Residuen des Modells auf Autokorrelation im Mean
% % results_errors_mean = lmtest1(errors,10); %H_0: es liegt keien Autokorrelation vor
% % % Es liegt keine Autokorrealtion mehr vor
% %
% % % % % Checke auf Heteroskedastizität der Residuen
% % results_errors_h = lmtest2(errors,10); %H_0: es liegt keine Autokorrelation vor !!Der Test quadriert die Residuen!!
% % % Es liegt Heteroskedastizität in den Residuen vor

% % % Fitte die verschiedenen GARCH_Modelle

% 1.GARCH-Gauss
% funktion liefert positive llf werte, desahalb muss mit 2 multpliziert
% werden und nicht mit -2
for i=1:5
    [parameters_GARCH_Gauss, likelihood_GARCH_Gauss(i), stderrors_GARCH_Gauss, robustSE_GARCH_Gauss, ht_GARCH_Gauss, scores_GARCH_Gauss, resid_GARCH_Gauss]=ar_multigarch_grm(x,1,0,1,'GARCH','NORMAL',i);
    BIC_GARCH_Gauss(i) = 2*likelihood_GARCH_Gauss(i)+2*log(t)*size(parameters_GARCH_Gauss,1);
end
[m,n] = min(BIC_GARCH_Gauss);
[parameters_GARCH_Gauss, likelihood_GARCH_Gauss, stderrors_GARCH_Gauss, robustSE_GARCH_Gauss, ht_GARCH_Gauss, scores_GARCH_Gauss, resid_GARCH_Gauss]=ar_multigarch_grm(x,1,0,1,'GARCH','NORMAL',n);
BIC_GARCH_Gauss = 2*likelihood_GARCH_Gauss+2*log(t)*size(parameters_GARCH_Gauss,1);
Tstatistic_GARCH_Gauss=parameters_GARCH_Gauss./diag(robustSE_GARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_GARCH_Gauss./sqrt(ht_GARCH_Gauss);
result_stderrors_GARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_GARCH_Gauss, siglevel_GARCH_Gauss, KolmH_GARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_GARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_GARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_GARCH_Gauss_U, siglevel_GARCH_Gauss_U, H_GARCH_Gauss_U]=kolmogorov(U_GARCH_Gauss,.05,'unifcdf',0,1);
U_GARCH_Gauss_mean = U_GARCH_Gauss-mean(U_GARCH_Gauss);
U_GARCH_Gauss_mean2 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^2;
U_GARCH_Gauss_mean3 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^3;
U_GARCH_Gauss_mean4 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_GARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);


% %2.AVGARCH-Gauss
% for i=1:5
%     [parameters_AVGARCH_Gauss, likelihood_AVGARCH_Gauss, stderrors_AVGARCH_Gauss, robustSE_AVGARCH_Gauss, ht_AVGARCH_Gauss, scores_AVGARCH_Gauss, resid_AVGARCH_Gauss, LLF_AVGARCH_Gauss(i)]=ar_multigarch_grm(x,1,0,1,'AVGARCH','NORMAL',i);
%     BIC_AVGARCH_Gauss(i) = -2*LLF_AVGARCH_Gauss(i)+2*log(t)*size(parameters_AVGARCH_Gauss,1);
% end
% [m,n] = min(BIC_AVGARCH_Gauss);
% [parameters_AVGARCH_Gauss, likelihood_AVGARCH_Gauss, stderrors_AVGARCH_Gauss, robustSE_AVGARCH_Gauss, ht_AVGARCH_Gauss, scores_AVGARCH_Gauss, resid_AVGARCH_Gauss, LLF_AVGARCH_Gauss]=ar_multigarch_grm(x,1,0,1,'AVGARCH','NORMAL',n);
% BIC_AVGARCH_Gauss = -2*LLF_AVGARCH_Gauss+2*log(t)*size(parameters_AVGARCH_Gauss,1);
% Tstatistic_AVGARCH_Gauss=parameters_AVGARCH_Gauss./diag(robustSE_AVGARCH_Gauss).^0.5;
% % H0: keine autocorrelation - in diesem Fall der quadrierten
% % standardisierten Residuen
% stdresid = resid_AVGARCH_Gauss./sqrt(ht_AVGARCH_Gauss);
% result_stderrors_AVGARCH_Gauss=lmtest2(stdresid,10);
% % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% % wird abgelehnt
% [statistic_AVGARCH_Gauss, siglevel_AVGARCH_Gauss, KolmH_AVGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
% U_AVGARCH_Gauss=zeros(size(stdresid,1),1);
% for i=1:size(stdresid,1)
%     U_AVGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
% end
% [statistic_AVGARCH_Gauss_U, siglevel_AVGARCH_Gauss_U, H_AVGARCH_Gauss_U]=kolmogorov(U_AVGARCH_Gauss,.05,'unifcdf',0,1);
% U_AVGARCH_Gauss_mean = U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss);
% U_AVGARCH_Gauss_mean2 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^2;
% U_AVGARCH_Gauss_mean3 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^3;
% U_AVGARCH_Gauss_mean4 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^4;
% % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
% [H_AVGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
BIC_AVGARCH_Gauss = 1e10;




% 3.NGARCH-Gauss
for i=1:5
    [parameters_NGARCH_Gauss, likelihood_NGARCH_Gauss, stderrors_NGARCH_Gauss, robustSE_NGARCH_Gauss, ht_NGARCH_Gauss, scores_NGARCH_Gauss, resid_NGARCH_Gauss, LLF_NGARCH_Gauss(i)]=ar_multigarch_grm(x,1,0,1,'NGARCH','NORMAL',i);
    BIC_NGARCH_Gauss(i) = -2*LLF_NGARCH_Gauss(i)+2*log(t)*size(parameters_NGARCH_Gauss,1);
end
[m,n] = min(BIC_NGARCH_Gauss);
[parameters_NGARCH_Gauss, likelihood_NGARCH_Gauss, stderrors_NGARCH_Gauss, robustSE_NGARCH_Gauss, ht_NGARCH_Gauss, scores_NGARCH_Gauss, resid_NGARCH_Gauss, LLF_NGARCH_Gauss]=ar_multigarch_grm(x,1,0,1,'NGARCH','NORMAL',n);
BIC_NGARCH_Gauss = -2*LLF_NGARCH_Gauss+2*log(t)*size(parameters_NGARCH_Gauss,1);
Tstatistic_NGARCH_Gauss=parameters_NGARCH_Gauss./diag(robustSE_NGARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_NGARCH_Gauss./sqrt(ht_NGARCH_Gauss);
result_stderrors_NGARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_NGARCH_Gauss, siglevel_NGARCH_Gauss, KolmH_NGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_NGARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_NGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_NGARCH_Gauss_U, siglevel_NGARCH_Gauss_U, H_NGARCH_Gauss_U]=kolmogorov(U_NGARCH_Gauss,.05,'unifcdf',0,1);
U_NGARCH_Gauss_mean = U_NGARCH_Gauss-mean(U_NGARCH_Gauss);
U_NGARCH_Gauss_mean2 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^2;
U_NGARCH_Gauss_mean3 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^3;
U_NGARCH_Gauss_mean4 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_NGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);



% 4.NAGARCH-Gauss
for i=1:5
    [parameters_NAGARCH_Gauss, likelihood_NAGARCH_Gauss, stderrors_NAGARCH_Gauss, robustSE_NAGARCH_Gauss, ht_NAGARCH_Gauss, scores_NAGARCH_Gauss, resid_NAGARCH_Gauss, LLF_NAGARCH_Gauss(i)]=ar_multigarch_grm(x,1,0,1,'NAGARCH','NORMAL',i);
    BIC_NAGARCH_Gauss(i) = -2*LLF_NAGARCH_Gauss(i)+2*log(t)*size(parameters_NAGARCH_Gauss,1);
end
[m,n] = min(BIC_NAGARCH_Gauss);
[parameters_NAGARCH_Gauss, likelihood_NAGARCH_Gauss, stderrors_NAGARCH_Gauss, robustSE_NAGARCH_Gauss, ht_NAGARCH_Gauss, scores_NAGARCH_Gauss, resid_NAGARCH_Gauss, LLF_NAGARCH_Gauss]=ar_multigarch_grm(x,1,0,1,'NAGARCH','NORMAL',n);
BIC_NAGARCH_Gauss = -2*LLF_NAGARCH_Gauss+2*log(t)*size(parameters_NAGARCH_Gauss,1);
Tstatistic_NAGARCH_Gauss=parameters_NAGARCH_Gauss./diag(robustSE_NAGARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_NAGARCH_Gauss./sqrt(ht_NAGARCH_Gauss);
result_stderrors_NAGARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_NAGARCH_Gauss, siglevel_NAGARCH_Gauss, KolmH_NAGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_NAGARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_NAGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_NAGARCH_Gauss_U, siglevel_NAGARCH_Gauss_U, H_NAGARCH_Gauss_U]=kolmogorov(U_NAGARCH_Gauss,.05,'unifcdf',0,1);
U_NAGARCH_Gauss_mean = U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss);
U_NAGARCH_Gauss_mean2 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^2;
U_NAGARCH_Gauss_mean3 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^3;
U_NAGARCH_Gauss_mean4 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_NAGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);



% % % 5.EGARCH-Gauss
% funktion liefert positive llf werte, desahalb muss mit 2 multpliziert
% werden und nicht mit -2
for i=1:5
    [parameters_EGARCH_Gauss, likelihood_EGARCH_Gauss(i), stderrors_EGARCH_Gauss, robustSE_EGARCH_Gauss, ht_EGARCH_Gauss, scores_EGARCH_Gauss, resid_EGARCH_Gauss]=ar_multigarch_grm(x,1,1,1,'EGARCH','NORMAL',i);
    BIC_EGARCH_Gauss(i) = 2*likelihood_EGARCH_Gauss(i)+2*log(t)*size(parameters_EGARCH_Gauss,1);
end
[m,n] = min(BIC_EGARCH_Gauss);
[parameters_EGARCH_Gauss, likelihood_EGARCH_Gauss, stderrors_EGARCH_Gauss, robustSE_EGARCH_Gauss, ht_EGARCH_Gauss, scores_EGARCH_Gauss, resid_EGARCH_Gauss]=ar_multigarch_grm(x,1,1,1,'EGARCH','NORMAL',n);
BIC_EGARCH_Gauss = 2*likelihood_EGARCH_Gauss+2*log(t)*size(parameters_EGARCH_Gauss,1);
Tstatistic_EGARCH_Gauss=parameters_EGARCH_Gauss./diag(robustSE_EGARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_EGARCH_Gauss./sqrt(ht_EGARCH_Gauss);
result_stderrors_EGARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_EGARCH_Gauss, siglevel_EGARCH_Gauss, KolmH_EGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_EGARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_EGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_EGARCH_Gauss_U, siglevel_EGARCH_Gauss_U, H_EGARCH_Gauss_U]=kolmogorov(U_EGARCH_Gauss,.05,'unifcdf',0,1);
U_EGARCH_Gauss_mean = U_EGARCH_Gauss-mean(U_EGARCH_Gauss);
U_EGARCH_Gauss_mean2 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^2;
U_EGARCH_Gauss_mean3 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^3;
U_EGARCH_Gauss_mean4 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_EGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);



% 6.TGARCH-Gauss
for i=1:5
    [parameters_TGARCH_Gauss, likelihood_TGARCH_Gauss, stderrors_TGARCH_Gauss, robustSE_TGARCH_Gauss, ht_TGARCH_Gauss, scores_TGARCH_Gauss, resid_TGARCH_Gauss, LLF_TGARCH_Gauss(i)]=ar_multigarch_grm(x,1,1,1,'TGARCH','NORMAL',i);
    BIC_TGARCH_Gauss(i) = -2*LLF_TGARCH_Gauss(i)+2*log(t)*size(parameters_TGARCH_Gauss,1);
end
[m,n] = min(BIC_TGARCH_Gauss);
[parameters_TGARCH_Gauss, likelihood_TGARCH_Gauss, stderrors_TGARCH_Gauss, robustSE_TGARCH_Gauss, ht_TGARCH_Gauss, scores_TGARCH_Gauss, resid_TGARCH_Gauss, LLF_TGARCH_Gauss]=ar_multigarch_grm(x,1,1,1,'TGARCH','NORMAL',n);
BIC_TGARCH_Gauss = -2*LLF_TGARCH_Gauss+2*log(t)*size(parameters_TGARCH_Gauss,1);
Tstatistic_TGARCH_Gauss=parameters_TGARCH_Gauss./diag(robustSE_TGARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_TGARCH_Gauss./sqrt(ht_TGARCH_Gauss);
result_stderrors_TGARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_TGARCH_Gauss, siglevel_TGARCH_Gauss, KolmH_TGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_TGARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_TGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_TGARCH_Gauss_U, siglevel_TGARCH_Gauss_U, H_TGARCH_Gauss_U]=kolmogorov(U_TGARCH_Gauss,.05,'unifcdf',0,1);
U_TGARCH_Gauss_mean = U_TGARCH_Gauss-mean(U_TGARCH_Gauss);
U_TGARCH_Gauss_mean2 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^2;
U_TGARCH_Gauss_mean3 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^3;
U_TGARCH_Gauss_mean4 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_TGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);


% 7.GJRGARCH-Gauss
for i=1:5
    [parameters_GJRGARCH_Gauss, likelihood_GJRGARCH_Gauss, stderrors_GJRGARCH_Gauss, robustSE_GJRGARCH_Gauss, ht_GJRGARCH_Gauss, scores_GJRGARCH_Gauss, resid_GJRGARCH_Gauss, LLF_GJRGARCH_Gauss(i)]=ar_multigarch_grm(x,1,1,1,'GJRGARCH','NORMAL',i);
    BIC_GJRGARCH_Gauss(i) = -2*LLF_GJRGARCH_Gauss(i)+2*log(t)*size(parameters_GJRGARCH_Gauss,1);
end
[m,n] = min(BIC_GJRGARCH_Gauss);
[parameters_GJRGARCH_Gauss, likelihood_GJRGARCH_Gauss, stderrors_GJRGARCH_Gauss, robustSE_GJRGARCH_Gauss, ht_GJRGARCH_Gauss, scores_GJRGARCH_Gauss, resid_GJRGARCH_Gauss, LLF_GJRGARCH_Gauss]=ar_multigarch_grm(x,1,1,1,'GJRGARCH','NORMAL',n);
BIC_GJRGARCH_Gauss = -2*LLF_GJRGARCH_Gauss+2*log(t)*size(parameters_GJRGARCH_Gauss,1);
Tstatistic_GJRGARCH_Gauss=parameters_GJRGARCH_Gauss./diag(robustSE_GJRGARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_GJRGARCH_Gauss./sqrt(ht_GJRGARCH_Gauss);
result_stderrors_GJRGARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_GJRGARCH_Gauss, siglevel_GJRGARCH_Gauss, KolmH_GJRGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_GJRGARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_GJRGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_GJRGARCH_Gauss_U, siglevel_GJRGARCH_Gauss_U, H_GJRGARCH_Gauss_U]=kolmogorov(U_GJRGARCH_Gauss,.05,'unifcdf',0,1);
U_GJRGARCH_Gauss_mean = U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss);
U_GJRGARCH_Gauss_mean2 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^2;
U_GJRGARCH_Gauss_mean3 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^3;
U_GJRGARCH_Gauss_mean4 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_GJRGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);



%
% 8.APGARCH-Gauss
for i=1:5
    [parameters_APGARCH_Gauss, likelihood_APGARCH_Gauss, stderrors_APGARCH_Gauss, robustSE_APGARCH_Gauss, ht_APGARCH_Gauss, scores_APGARCH_Gauss, resid_APGARCH_Gauss, LLF_APGARCH_Gauss(i)]=ar_multigarch_grm(x,1,1,1,'APGARCH','NORMAL',i);
    BIC_APGARCH_Gauss(i) = -2*LLF_APGARCH_Gauss(i)+2*log(t)*size(parameters_APGARCH_Gauss,1);
end
[m,n] = min(BIC_APGARCH_Gauss);
[parameters_APGARCH_Gauss, likelihood_APGARCH_Gauss, stderrors_APGARCH_Gauss, robustSE_APGARCH_Gauss, ht_APGARCH_Gauss, scores_APGARCH_Gauss, resid_APGARCH_Gauss, LLF_APGARCH_Gauss]=ar_multigarch_grm(x,1,1,1,'APGARCH','NORMAL',n);
BIC_APGARCH_Gauss = -2*LLF_APGARCH_Gauss+2*log(t)*size(parameters_APGARCH_Gauss,1);
Tstatistic_APGARCH_Gauss=parameters_APGARCH_Gauss./diag(robustSE_APGARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_APGARCH_Gauss./sqrt(ht_APGARCH_Gauss);
result_stderrors_APGARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_APGARCH_Gauss, siglevel_APGARCH_Gauss, KolmH_APGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_APGARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_APGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_APGARCH_Gauss_U, siglevel_APGARCH_Gauss_U, H_APGARCH_Gauss_U]=kolmogorov(U_APGARCH_Gauss,.05,'unifcdf',0,1);
U_APGARCH_Gauss_mean = U_APGARCH_Gauss-mean(U_APGARCH_Gauss);
U_APGARCH_Gauss_mean2 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^2;
U_APGARCH_Gauss_mean3 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^3;
U_APGARCH_Gauss_mean4 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_APGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);



% 9.ALLGARCH-Gauss
for i=1:5
    [parameters_ALLGARCH_Gauss, likelihood_ALLGARCH_Gauss, stderrors_ALLGARCH_Gauss, robustSE_ALLGARCH_Gauss, ht_ALLGARCH_Gauss, scores_ALLGARCH_Gauss, resid_ALLGARCH_Gauss, LLF_ALLGARCH_Gauss(i)]=ar_multigarch_grm(x,1,1,1,'ALLGARCH','NORMAL',i);
    BIC_ALLGARCH_Gauss(i) = -2*LLF_ALLGARCH_Gauss(i)+2*log(t)*size(parameters_ALLGARCH_Gauss,1);
end
[m,n] = min(BIC_ALLGARCH_Gauss);
[parameters_ALLGARCH_Gauss, likelihood_ALLGARCH_Gauss, stderrors_ALLGARCH_Gauss, robustSE_ALLGARCH_Gauss, ht_ALLGARCH_Gauss, scores_ALLGARCH_Gauss, resid_ALLGARCH_Gauss, LLF_ALLGARCH_Gauss]=ar_multigarch_grm(x,1,1,1,'ALLGARCH','NORMAL',n);
BIC_ALLGARCH_Gauss = -2*LLF_ALLGARCH_Gauss+2*log(t)*size(parameters_ALLGARCH_Gauss,1);
Tstatistic_ALLGARCH_Gauss=parameters_ALLGARCH_Gauss./diag(robustSE_ALLGARCH_Gauss).^0.5;
% H0: keine autocorrelation - in diesem Fall der quadrierten
% standardisierten Residuen
stdresid = resid_ALLGARCH_Gauss./sqrt(ht_ALLGARCH_Gauss);
result_stderrors_ALLGARCH_Gauss=lmtest2(stdresid,10);
% H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
% Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
% wird abgelehnt
[statistic_ALLGARCH_Gauss, siglevel_ALLGARCH_Gauss, KolmH_ALLGARCH_Gauss]=kolmogorov(stdresid,.05,'norm_cdf');
U_ALLGARCH_Gauss=zeros(size(stdresid,1),1);
for i=1:size(stdresid,1)
    U_ALLGARCH_Gauss(i,:)=norm_cdf(stdresid(i),0,1);
end
[statistic_ALLGARCH_Gauss_U, siglevel_ALLGARCH_Gauss_U, H_ALLGARCH_Gauss_U]=kolmogorov(U_ALLGARCH_Gauss,.05,'unifcdf',0,1);
U_ALLGARCH_Gauss_mean = U_ALLGARCH_Gauss-mean(U_ALLGARCH_Gauss);
U_ALLGARCH_Gauss_mean2 = (U_ALLGARCH_Gauss-mean(U_ALLGARCH_Gauss)).^2;
U_ALLGARCH_Gauss_mean3 = (U_ALLGARCH_Gauss-mean(U_ALLGARCH_Gauss)).^3;
U_ALLGARCH_Gauss_mean4 = (U_ALLGARCH_Gauss-mean(U_ALLGARCH_Gauss)).^4;
% H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
[H_ALLGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);

BIC_gesamt_Gauss = [BIC_GARCH_Gauss;BIC_AVGARCH_Gauss;BIC_NGARCH_Gauss;BIC_NAGARCH_Gauss;BIC_EGARCH_Gauss;BIC_TGARCH_Gauss;BIC_GJRGARCH_Gauss;...
    BIC_APGARCH_Gauss;BIC_ALLGARCH_Gauss;];
optGARCH_Gauss = find(BIC_gesamt_Gauss == min(BIC_gesamt_Gauss))
BIC_NORMAL = BIC_gesamt_Gauss(optGARCH_Gauss);
% % Stelle die U's grafisch dar
% figure(1)
% subplot(3,3,1); hist(U_GARCH_Gauss,40); title('GARCH Gauss')
% subplot(3,3,2); 0;title('AVGARCH_Gauss')
% subplot(3,3,3); hist(U_NGARCH_Gauss,40); title('NGARCH_Gauss')
% subplot(3,3,4); hist(U_NAGARCH_Gauss,40); title('NAGARCH_Gauss')
% subplot(3,3,5); hist(U_EGARCH_Gauss,40);title('EGARCH_Gauss')
% subplot(3,3,6); hist(U_TGARCH_Gauss,40); title('TGARCH_Gauss')
% subplot(3,3,7); hist(U_GJRGARCH_Gauss,40); title('GJRGARCH_Gauss')
% subplot(3,3,8); hist(U_APGARCH_Gauss,40);title('APGARCH_Gauss')

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% !!! optimales Modell: AR(2)-GARCH-Gauss
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
