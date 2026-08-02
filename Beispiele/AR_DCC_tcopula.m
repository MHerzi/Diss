% Liest die Daten ein

clear all;

% % Lese den HFRXEMN ein
cd G:\Daten
y = xlsread('Hedge_HFRX_20090611',1);
daten=[price2ret(y(:,2:3)) price2ret(y(:,14))];

[t,k]=size(daten);

archP=1;
garchQ=1;
dccP=1;
dccQ=1;
dccG=1;
asymmG=1;

if isempty(archP)
    archP=ones(1,k);
elseif length(archP)==1
    archP=ones(1,k)*archP;
end

if isempty(garchQ)
    garchQ=ones(1,k);
elseif length(garchQ)==1
    garchQ=ones(1,k)*garchQ;
end

if isempty(asymmG)
    asymmG=ones(1,k);
elseif length(garchQ)==1
    asymmG=ones(1,k)*garchQ;
end

% Marginalmodelle
% HFRXEMN
[parameters_HFRXEMN, likelihood_HFRXEMN, stderrors_HFRXEMN, robustSE_HFRXEMN, ht_HFRXEMN, scores_HFRXEMN, resid_HFRXEMN]=ar_multigarch_grm(daten(:,1),1,0,1,'GARCH','GED',1);
ht1_HFRXEMN = parameters_HFRXEMN(1) + parameters_HFRXEMN(2)*resid_HFRXEMN.^2 + parameters_HFRXEMN(3)*ht_HFRXEMN;
rt1_HFRXEMN = parameters_HFRXEMN(end-2) + parameters_HFRXEMN(end-1)*resid_HFRXEMN;
% DJW1
[parameters_DJW1, likelihood_DJW1, stderrors_DJW1, robustSE_DJW1, ht_DJW1, scores_DJW1, resid_DJW1]=ar_multigarch_grm(daten(:,2),1,1,1,'EGARCH','NORMAL',1);
ht1_DJW1 = exp(parameters_DJW1(1) + parameters_DJW1(2)*abs(resid_DJW1)./sqrt(ht_DJW1) + parameters_DJW1(3)*resid_DJW1./sqrt(ht_DJW1)  + parameters_DJW1(4)*log(ht_DJW1)); 
rt1_DJW1 = parameters_DJW1(end-1) + parameters_DJW1(end)*resid_DJW1;
% MerrillGlobal
[parameters_MerrillGlobal, likelihood_MerrillGlobal, stderrors_MerrillGlobal, robustSE_MerrillGlobal, ht_MerrillGlobal, scores_MerrillGlobal, resid_MerrillGlobal]=ar_multigarch_grm(daten(:,3),1,0,1,'GARCH','STUDENTST',1);
ht1_MerrillGlobal = parameters_MerrillGlobal(1) + parameters_MerrillGlobal(2)*resid_MerrillGlobal.^2 + parameters_MerrillGlobal(3)*ht_MerrillGlobal;
rt1_MerrillGlobal = parameters_MerrillGlobal(end-2) + parameters_MerrillGlobal(end-1)*resid_MerrillGlobal;
% % % EURUSD
% [parameters_EURUSD, likelihood_EURUSD, stderrors_EURUSD, robustSE_EURUSD, ht_EURUSD, scores_EURUSD, resid_EURUSD]=ar_multigarch_grm(daten(:,3),1,0,1,'GARCH','GED',1);
% ht1_EURUSD = parameters_EURUSD(1) + parameters_EURUSD(2)*resid_EURUSD.^2 + parameters_EURUSD(3)*ht_EURUSD;
% rt1_EURUSD = parameters_EURUSD(end-1) + parameters_EURUSD(end-1)*resid_EURUSD;

resid = [resid_HFRXEMN resid_DJW1 resid_MerrillGlobal ]; %resid_EURUSD
stdresid = [resid(:,1)./sqrt(ht_HFRXEMN) resid(:,2)./sqrt(ht_DJW1) resid(:,3)./sqrt(ht_MerrillGlobal)];%resid_EURUSD./sqrt(ht_EURUSD)

[t,k] = size(resid);
u1 = gedcdf(stdresid(:,1),parameters_HFRXEMN(end));
u2 = normcdf(stdresid(:,2),0,1);
u3 = tcdf(stdresid(:,3),parameters_MerrillGlobal(end));
% u4 = gedcdf(stdresid(:,4),parameters_EURUSD(end));


% Festlegen der Optimierungseinstellungen
options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6,'Algorithm','interior-point','Hessian','bfgs','FunValCheck','on','MaxFunEvals',10000);

% Hier wird die t-Verteilung der Innovations angenommen
% 8a tri-StudentT
[kappa_static_t,nuhat_static_t,nuci_static_t] = copulafit('t',[u1, u2, u3],'Options',options);
% time-varying tri t-copula
lower = [0+(1e-10); 0+(1e-10); 2.01];
upper = [1-(1e-10); 1-(1e-10); 100];
theta0 = [.01;.9; 10];

% Constraints, damit a+b < 1
A = [1 1 0];
b = 1-1e-10;

% Schätzen der Funktionen
% DCC
[ kappa_DCC LL_DCC] = fmincon('multi_tCopula_tvp1_DCCRt',theta0,A,b,[],[],lower,upper,'DCC_nonlincon',options,[u3, u2, u1],dccP,dccQ,k);% In dem Paper von DCC(2002) wir m = k gesetzt;
[LL_DCC Rt_DCC] = multi_tCopula_tvp1_DCCRt(kappa_DCC, [u3, u1, u2],dccP, dccQ,k);
% 
% Runden, auf die fünfte Nachkommastelle, da sonst auf der Hauptdiagonalen
% Elemente stehen die ~=1 sind
RtDCC = round(Rt_DCC*1e5)*1e-5; 

t=size(RtDCC,3);

% generiere Zufallszahlen
N=1000;
copula='StudentT';
random = zeros(N,t*k);
h=1;
for i=1:t
    random(:,h:(h+(k-1))) = copularnd('t',RtDCC(:,:,i),kappa_DCC(end),N);
    h=h+k;
end

% Für den Backtest ist der letzte Forecast von t-1 auf t; in t ist die
% letzte PuL

% Wandle die Copula-Zufallszahlen U(0,1) wieder in Returns zurück
h=1;
for j=1:k
    for i = 1:k:t*k
        crnd{j}(:,h) = tinv(random(:,i),kappa_DCC(end));
        h=h+1;
    end
    h=1;
end
        
clear random

% generiere Returns aus den Zufallsvaribalen (für den Zeitpunkt t) und
% wähle den jeweiligen VaR aus - je nach Konfidenzniveau

MeanForecast = [rt1_HFRXEMN rt1_DJW1 rt1_MerrillGlobal]; %rt1_EURUSD
SigmaForecast = sqrt([ht1_HFRXEMN ht1_DJW1 ht1_MerrillGlobal]); %rt1_EURUSD

n=N;

MeanForecast=MeanForecast(size(MeanForecast,1)-t+1:end,:);
SigmaForecast=SigmaForecast(size(SigmaForecast,1)-t+1:end,:);

rnd_ret = cell(1,k);
for i=1:k
    rnd_ret{1,i} = zeros(n,t);
end
VaR991 = cell(1,k);
for i=1:k
    VaR991{1,i} = zeros(t,1);
end
VaR951 = cell(1,k);
for i=1:k
    VaR951{1,i} = zeros(t,1);
end
VaR11 = cell(1,k);
for i=1:k
    VaR11{1,i} = zeros(t,1);
end
VaR51 = cell(1,k);
for i=1:k
    VaR51{1,i} = zeros(t,1);
end

for i=1:k
    for j=1:t
        rnd_ret{1,i}(:,j) = MeanForecast(j,i) + SigmaForecast(j,i)*crnd{i}(:,j);
    end
    rnd_ret{1,i}=sort(rnd_ret{1,i});
end

for i=1:k
    for j=1:t
        VaR991{i}(j) = rnd_ret{i}(n*0.01,j);
        VaR951{i}(j) = rnd_ret{i}(n*0.05,j);
        VaR11{i}(j)  = rnd_ret{i}(n*0.99,j);
        VaR51{i}(j)  = rnd_ret{i}(n*0.95,j);
    end
end
% Bilde Portfolio VaR
Port991 = zeros(t,1);
Port951 = zeros(t,1);
Port11 = zeros(t,1);
Port51 = zeros(t,1);
for i=1:k
    Port991 = Port991+(1/k)*VaR991{i};
    Port951 = Port951+(1/k)*VaR951{i};
    Port11 = Port11+(1/k)*VaR11{i};
    Port51 = Port51+(1/k)*VaR51{i};
end

% Berechne die wahre PuL
% Es gehen Beobachtungen durch die AR-Schätzung verloren (in diesem Fall
% eine)
true_PuL = zeros(t,1);
for i=1:k
    true_PuL = true_PuL + 1/k*daten(size(daten,1)-t+1:end,i);
end

% VaR (95/1)
figure
a=area(Port951(:,1));
set(a,'FaceColor',[1 1 0])
hold on
b=area(Port51(:,1));
set(b,'FaceColor',[1 1 0])
hold on
plot(true_PuL,'.')
title('AGDCC-VaR (95/1) 0.25*HFRXEMN + 0.25*DJW1 + 0.25*EURUSD + 0.25*MerrillGlobal')

% VaR (99/1)
figure
a=area(Port991(:,1));
set(a,'FaceColor',[1 1 0])
hold on
b=area(Port11(:,1));
set(b,'FaceColor',[1 1 0])
hold on
plot(true_PuL,'.')
title('AGDCC-VaR (99/1) 0.25*HFRXEMN + 0.25*DJW1 + 0.25*EURUSD + 0.25*MerrillGlobal')

% Zähel die VaR-Verletzungen (nur left tail), um die Modellgüte zu validieren
VaRviol991=zeros(t,1);
for i = 1:t-1
    if Port991(i) > true_PuL(i)
        VaRviol991(i) =  1;
    else VaRviol991(i) =  0;
    end
end
SumVaRviol991 = sum(VaRviol991);
opt_Verletzung = round((t-1)/100);
% Bei 1 hätte man das optimale VaR-Modell
VaRviol991per = SumVaRviol991./opt_Verletzung;

VaRviol951=zeros(t,1);
for i = 1:t-1
    if Port951(i) > true_PuL(i)
        VaRviol951(i) =  1;
    else VaRviol951(i) =  0;
    end
end
SumVaRviol951 = sum(VaRviol951);
% Bei 1 hätte man das optimale VaR-Modell
VaRviol951per = SumVaRviol951./opt_Verletzung;

% Referenzen Wilmott, P. (2006). Wilmott on Quantitative Finance 2nd ed.

% Tests nach Christoffersen ("Backtesting"), S.4,
%
% Simple means (unconditional coverage) test H0 : E(It) = p (im Mittel treten p-Verletzungen auf)
% Formel: MT sqrt(T)*(pi - p)/sqrt(VaR(I_t) ist Normalverteilt (0,1)
% I_t: Sequenz von Nullen und Einsen - die hit sequence (Eins: der Verlust der PuL ist größer
% als der VaR)
% pi : sample mean der hit-sequence
% Bilde den Sample Average der hit-sequence
T1 = SumVaRviol991;
T = size(Port991,1);
mean_hit = T1/T;
p = 0.01; %Konfidenzniveau
var_It = var(Port991)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.99,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
Vergleich_test1 = [MT norm_krit];

% Likelihood ratio test
% Nullhypothese pi = p, also erwartete Wert der hit-Sequenz (pi) soll dem
% coverage ratio (p) entsprechen
T0 = T-T1;
% optimierte Likelihood
Like1 = (1-T1/T)^T0*(T1/T)^T1;
Like2 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR = -2*log(Like2/Like1);
% kritischer Wert der Chi^2-Verteilung
krit_chi = chi2inv(0.99,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
Vergleich_test2 = [LR krit_chi];



