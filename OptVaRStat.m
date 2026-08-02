function Output = OptVaRStat(VaR991,daten,Spec)

% berechne die verschiedenen Statistiken des normalen VaR's auch für das
% potimierte VaR - nur für 99/1 VaR

forecastP = Spec.ForecastNumb;
t2 =  forecastP;
k = size(daten,2);
% berechne die wahre PuL
true_PuL=zeros(size(VaR991,1),1);
for i=1:k
    true_PuL = true_PuL + 1/k*exp(daten(end-t2+1:end,i)); %bei Verwendung von Originaldaten muss für die einperiodige Vorhersage
end
true_PuL = log(true_PuL);
Output.truePuL = true_PuL;

% Grafik
figure
a=area(VaR991(:,1)); %VaR ist als positive Zahl definiert
set(a,'FaceColor',[1 1 0])
hold on
for i=1:size(VaR991,1)
    if -true_PuL(i)<0
        true_PuL(i)=0;
    end
end
plot(-true_PuL(:,1),'.')
title('VaR (99/1)')

VaRviol991=zeros(size(VaR991,1),1);
for i = 1:size(VaR991,1)
    if VaR991(i) < -true_PuL(i)
        VaRviol991(i) =  1;
    else VaRviol991(i) =  0;
    end
end
SumVaRviol991 = sum(VaRviol991);
Output.SumVaRviol991 = sum(VaRviol991);


% Referenzen Wilmott, P. (2006). Wilmott on Quantitative Finance 2nd ed.

% ------------------------------------------------------------------------
% -------------------------------------------------------------------------
% VaR-Tests
% Tests nach Christoffersen ("Backtesting"), S.4,
%
% %%%%% unconditional Coverage Test %%%%%%%%%%%
% Simple means (unconditional coverage) test H0 : E(It) = p (im Mittel treten p-Verletzungen auf)
% Formel: MT sqrt(T)*(pi - p)/sqrt(VaR(I_t) ist Normalverteilt (0,1)
% I_t: Sequenz von Nullen und Einsen - die hit sequence (Eins: der Verlust der PuL ist größer
% als der VaR)
% pi : sample mean der hit-sequence
% Bilde den Sample Average der hit-sequence
T1 = SumVaRviol991;
T = size(VaR991,1);
mean_hit = T1/T;
p = 0.01; %Konfidenzniveau
var_It = var(VaRviol991)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 1-normcdf(MT,0,1);
Output.UncondCoverage991_MT = [MT norm_krit pval];

% % für VaR95-1
% T1 = SumVaRviol951;
% T = size(Port951,1);
% mean_hit = T1/T;
% p = 0.05; %Konfidenzniveau
% var_It = var(VaRviol951)/T;
% MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% % kritischer Wert der Normalverteilung
% norm_krit = norminv(0.90,0,1);
% % Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% % Teststatistik
% pval = 1-normcdf(MT,0,1);
% Output.UncondCoverage951_MT = [MT norm_krit pval];
% 
% % für VaR90-1
% T1 = SumVaRviol901;
% T = size(Port901,1);
% mean_hit = T1/T;
% p = 0.1; %Konfidenzniveau
% var_It = var(VaRviol901)/T;
% MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% % kritischer Wert der Normalverteilung
% norm_krit = norminv(0.90,0,1);
% % Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% % Teststatistik
% pval = 1-normcdf(MT,0,1);
% Output.UncondCoverage901_MT = [MT norm_krit pval];
% 
% % für VaR999-1
% T1 = SumVaRviol9991;
% T = size(Port9991,1);
% mean_hit = T1/T;
% p = 0.001; %Konfidenzniveau
% var_It = var(VaRviol9991)/T;
% MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% % kritischer Wert der Normalverteilung
% norm_krit = norminv(0.90,0,1);
% % Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% % Teststatistik
% pval = 1-normcdf(MT,0,1);
% Output.UncondCoverage9991_MT = [MT norm_krit pval];

% -----------------------------------------------------------------------
% Likelihood ratio test
% Nullhypothese pi = p, also erwartete Wert der hit-Sequenz (pi) soll dem
% coverage ratio (p) entsprechen
T1 = SumVaRviol991;
T = size(VaR991,1);
T0 = T-T1;
p=0.01;
% optimierte Likelihood
Like1_991 = (1-T1/T)^T0*(T1/T)^T1;
Like2_991 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR991 = -2*log(Like2_991/Like1_991);
% kritischer Wert der Chi^2-Verteilung
krit_chi991 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR991,1);
Output.Independence991_LR = [LR991 krit_chi991 pval];

% für VaR 951
% T1 = SumVaRviol951;
% T = size(Port951,1);
% T0 = T-T1;
% p=0.05;
% % optimierte Likelihood
% Like1_951 = (1-T1/T)^T0*(T1/T)^T1;
% Like2_951 = (1-p)^T0*p^T1;
% % Likelihood Ratio Teststatistik
% LR951 = -2*log(Like2_951/Like1_951);
% % kritischer Wert der Chi^2-Verteilung
% krit_chi951 = chi2inv(0.90,1);
% % Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% % Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
% pval = 1-chi2cdf(LR951,1);
% Output.Independence951_LR = [LR951 krit_chi951 pval];
% 
% 
% % für VaR 901
% T1 = SumVaRviol901;
% T = size(Port901,1);
% T0 = T-T1;
% p=0.1;
% % optimierte Likelihood
% Like1_901 = (1-T1/T)^T0*(T1/T)^T1;
% Like2_901 = (1-p)^T0*p^T1;
% % Likelihood Ratio Teststatistik
% LR901 = -2*log(Like2_901/Like1_901);
% % kritischer Wert der Chi^2-Verteilung
% krit_chi901 = chi2inv(0.90,1);
% % Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% % Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
% pval = 1-chi2cdf(LR901,1);
% Output.Independence901_LR = [LR901 krit_chi901 pval];
% 
% % für VaR 9991
% T1 = SumVaRviol9991;
% T = size(Port9991,1);
% T0 = T-T1;
% p=0.1;
% % optimierte Likelihood
% Like1_9991 = (1-T1/T)^T0*(T1/T)^T1;
% Like2_9991 = (1-p)^T0*p^T1;
% % Likelihood Ratio Teststatistik
% LR9991 = -2*log(Like2_9991/Like1_9991);
% % kritischer Wert der Chi^2-Verteilung
% krit_chi9991 = chi2inv(0.90,1);
% % Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% % Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
% pval = 1-chi2cdf(LR9991,1);
% Output.Independence9991_LR = [LR9991 krit_chi9991 pval];

% -----------------------------------------------------------------------
% Ljung-Box (independence testing)
% siehe(Jin (2009), "Large Portfolio Risk Management with Dynamic Copulas)
% H0: model fit is adequate
[H_991,pValue_991,Qstat_991,CriticalValue_991] = lbqtest(VaRviol991,5,.10,5);
Output.LB991(1) = Qstat_991;
Output.LB991(2) = CriticalValue_991;
Output.LB991(3) = pValue_991;
% % für VaR 951
% [H_951,pValue_951,Qstat_951,CriticalValue_951] = lbqtest(VaRviol951,5,.10,5);
% Output.LB951(1) = Qstat_951;
% Output.LB951(2) = CriticalValue_951;
% Output.LB951(3) = pValue_951;
% % für VaR 901
% [H_901,pValue_901,Qstat_901,CriticalValue_901] = lbqtest(VaRviol901,5,.10,5);
% Output.LB901(1) = Qstat_901;
% Output.LB901(2) = CriticalValue_901;
% Output.LB901(3) = pValue_901;
% % für VaR 9991
% [H_9991,pValue_9991,Qstat_9991,CriticalValue_9991] = lbqtest(VaRviol9991,5,.10,5);
% Output.LB951(1) = Qstat_9991;
% Output.LB951(2) = CriticalValue_9991;
% Output.LB9991(3) = pValue_9991;
% 
% 
% % lower partial moment
% ELPM991 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR991{k},1)
%         ELPM991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR991{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM991 = ELPM991;
% 
% ELPM951 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR951{k},1)
%         ELPM991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR951{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM951 = ELPM951;
% 
% ELPM901 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR901{k},1)
%         ELPM901{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR901{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM901 = ELPM901;
% 
% ELPM9991 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR9991{k},1)
%         ELPM9991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR9991{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM9991 = ELPM9991;
