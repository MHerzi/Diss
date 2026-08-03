

% Example estimates VaR and Christoffersen's HIT-ratios for
% CopulaVine Functions; univariate GARH models are estimated with ever
% incrreasing information
%
% USAGE:  [VaR] = VineVaR(StrOutput,CopulaSpec,T)
%
% INPUT:
%         data:        t x k array of return series
%         Output:      Structure from fitCopulaVine_grm.m
%         CopulaSpec:  Structure from  setCopulaVineLLinputs_grm.m
%         T:           # of simulated variables
%
% OUTPUT:
%         VaR:         Structure containing estimatred VaR's
%         HIT:         Structure containing Christoffersen's HIT-ratios


% ------------------------------------------------------------------------

% -----------------------------------------------------------------------
const=1;
arlag=1;
% estimate marginal models
[GARCHOutput] = AR_MarginalModel(daten,const,arlag);

% bring all GARCHOutput vectors to same length
k=size(daten,2);
for i=1:k
    maxarlag(i)=GARCHOutput{i}.arlag;
    arlag(i) = GARCHOutput{i}.arlag;
end
maxarlag=max(maxarlag);

    if GARCHOutput{i}.arlag<maxarlag
        ardiff(i) = maxarlag-GARCHOutput{i}.arlag;
    else ardiff(i) = 0;
    end
    ht_new(:,i) = GARCHOutput{i}.ht(ardiff(i)+1:end);
    resid_new(:,i) = GARCHOutput{i}.Innovations(ardiff(i)+1:end);
    stdresid(:,i) = resid_new(:,i)./sqrt(ht_new(:,i));
    scores_new{i} = GARCHOutput{i}.Scores(ardiff(i)+1:end,:);
for i=1:k
    daten_new(:,i) = daten(maxarlag+1:end,i);
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% estimate GARCH models with timey varying window, where information is
% alway increasing
% forecastP: peridos to be forecasted
forecastP = 40;
start=3000; % # of data used to estimate the first GARCH
const=1;
[t,k]=size(stdresid);
r=1;
for i=start:forecastP:t
    [GARCHOutput_add{r}] = AR_MarginalModel_add(daten(1:i,:),const,GARCHOutput);
    r=r+1;
end
% estimate residuals, variances and residuals for every pair of estimated
% coeffcients
% AR and ht will be one period forecasts
r=1;
for i=start:forecastP:t-forecastP
    [AR_pred_all{r},ht_pred_all{r}, resid_pred_all{r}] = ar_univariate_snoop_forecast1P(daten(1:i+forecastP+maxarlag-1,:),GARCHOutput_add{r},const);
    r=r+1;
end
% all vectors need to have same length
for i=1:k-1
    for j=1:size(daten,2)
    AR_pred_all{i}{j} = AR_pred_all{i}{j}(ardiff(j)+1:end);
    ht_pred_all{i}{j} = ht_pred_all{i}{j}(ardiff(j)+1:end);
    resid_pred_all{i}{j} = resid_pred_all{i}{j}(ardiff(j)+1:end);
    end
end
% make one time series out of different estimated series
for i=1:k
    AR_pred(:,i)=AR_pred_all{1}{i}';
    ht_pred(:,i)=ht_pred_all{1}{i};
    resid_pred(:,i)=resid_pred_all{1}{i};
end
g=2;
for j=start+1+forecastP:forecastP:t-forecastP
    for h=1:k
        AR_pred(j:j+forecastP-1,h) = AR_pred_all{g}{h}(end-forecastP+1:end)';
        ht_pred(j:j+forecastP-1,h) = ht_pred_all{g}{h}(end-forecastP+1:end)';
        resid_pred(j:j+forecastP-1,h) = resid_pred_all{g}{h}(end-forecastP+1:end)';
    end
    g=g+1;
end
% make residuals (from 1 period predixtion) to unif(0,1) variables
h=1;
for j=start+forecastP:forecastP:size(AR_pred,1)
    for i=1:k
        if strcmp(GARCHOutput_add{h}{i}.dist,'GAUSS') == 1
            U_pred(1:j,i) = normcdf(resid_pred(1:j,i)./sqrt(ht_pred(:,i)));
        elseif strcmp(GARCHOutput_add{h}{i}.dist,'STUDENTST') == 1
            U_pred(1:j,i) = tcdf(resid_pred(1:j,i)./sqrt(ht_pred(1:j,i)),GARCHOutput_add{h}{i}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput_add{h}{i}.dist,'GED') == 1
            U_pred(1:j,i) = ged_icdf(resid_pred(1:j,i)./sqrt(ht_pred(1:j,i)),GARCHOutput_add{h}{i}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput_add{h}{i}.dist,'SKEWT') == 1
            U_pred(1:j,i) = tdis_cdf(resid_pred(1:j,i)./sqrt(ht_pred(1:j,i)),GARCHOutput_add{h}{i}.ParamsGARCH(end-1));
        end
    end
    h=h+1;
end

% ------------------------------------------------------------------------
% estimate vine-copula
% vine will be only estimated for sample less than out-of-sample
% forecasting sample
[LogL, VineOutput, CopulaSpec]=fitCopulaVine_grm(U_pred(1:start,:));
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% % use coefficients etsimates from first sample to calculate time-varying
% correlation for the whole sample; write coefficients as a vector vor use
% with CopulaVineLL_grm.m
h=k-1;
theta = VineOutput.VineParams{1}{1};
for i=1:k-1
    if i==1
        for c = 2:h
            theta = [theta; VineOutput.VineParams{i,c}{1}];
        end
    else
        for c = 1:h
            theta = [theta; VineOutput.VineParams{i,c}{1}];
        end
    end
    h=h-1;
end
% use predicted residuals tranformed to unif(0,1) to estimate Rt
[LogL,Rt] = CopulaVineLL_grm(theta,U_pred,CopulaSpec);
% --------------------------------------------------------------------

% -------------------------------------------------------------------
% Simulation Algorithm
% T: # of variables to be simulated
T=1000;
corrspec=CopulaSpec.corrspec;
[t,k]=size(U_pred);
Rt_new=cell(k-1,k-1);
Sim=cell(k-1,k-1);
if strcmp(corrspec,'Patton') == 1 || strcmp(corrspec,'DCC') == 1 || strcmp(corrspec,'TVC') == 1
    for h=1:t
            c=k-1;
        for i=1:.5*(k*(k-1))
            for j=1:c
                Rt_new{i,j} = Rt{i,j}(2,1,h);
            end
            c=c-1;
        end
        [Sim{h}] = SimVine(T,CopulaSpec,VineOutput,Rt_new);
    end
end
    

% -----------------------------------------------------------------------
%               VaR-calculation
% ----------------------------------------------------------------------
[t,k] = size(U_pred);
invdata = cell(t,1);
% ------------------------------------------------------------------------
% bestimmte stdresids aus den unif(0,1) daten; achte dabei darauf, dass die
% jeweiligen Intervalle mit den Intervallen der geschtzten GARCH-Modelle
% übereinstimmen

% für die erste Periode - bzw. das erste GARCH-Modell
h=1;
for i=1:start
    for j=1:k
        if strcmp(GARCHOutput_add{h}{j}.dist,'Gaussian') == 1;
            invdata{i}(:,j) = norminv(Sim{i}(:,j));
        elseif strcmp(GARCHOutput_add{h}{j}.dist,'STUDENTST') == 1;
            invdata{i}(:,j) = tinv(Sim{h}(:,j),GARCHOutput_add{h}{j}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput_add{h}{j}.dist,'GED') == 1;
            invdata{i}(:,j) = gedinv(Sim{h}(:,j),GARCHOutput_add{h}{j}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput_add{h}{j}.dist,'SKEWT') == 1;
            invdata{i}(:,j) = skewtdis_inv(Sim{h}(:,j),GARCHOutput_add{h}{j}.ParamsGARCH(end-1),GARCHOutput{j}.ParamsGARCH(end));
        end
    end
end
% für alle nachfolgenden Periode, bzw. GARCH-Modelle
h=2;
g=1;
for i=start+1:t
    for j=1:k
        if strcmp(GARCHOutput_add{h}{j}.dist,'Gaussian') == 1;
            invdata{i}(:,j) = norminv(Sim{i}(:,j));
        elseif strcmp(GARCHOutput_add{h}{j}.dist,'STUDENTST') == 1;
            invdata{i}(:,j) = tinv(Sim{h}(:,j),GARCHOutput_add{h}{j}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput_add{h}{j}.dist,'GED') == 1;
            invdata{i}(:,j) = gedinv(Sim{h}(:,j),GARCHOutput_add{h}{j}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput_add{h}{j}.dist,'SKEWT') == 1;
            invdata{i}(:,j) = skewtdis_inv(Sim{h}(:,j),GARCHOutput_add{h}{j}.ParamsGARCH(end-1),GARCHOutput{j}.ParamsGARCH(end));
        end
    end
    %     zähle h alle forecastP um eins hoch, um die Koeffizienten des
    %     nächsten GARCH-Modells zu verwenden
    if g == forecastP+1
        h=h+1;
        g=1;
    end
       g=g+1;
end
% -------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Generiere aus den standardisierten Returns wieder nicht-standardisierte
% benutze dazu den forecast für den Mittelwert und die Varianz
% write AR_pred results in one cell
VaR_ret=cell(t,1);
% generate returns for VaR using predicted mean and predicted vola
for j=1:t
    for i=1:T
        VaR_ret{j}(i,:) = AR_pred(j,:)+sqrt(ht_pred(j,:)).*invdata{j}(i,:);
    end
      VaR_ret{j} = sort(VaR_ret{j});
end

VaR991= cell(t,1);
VaR951= cell(t,1);
VaR51= cell(t,1);
VaR11= cell(t,1);
% Var 991 951 51 and 11
% j represents every day
for j = 1:t
    numb = T*0.01;
    VaR991{j} = VaR_ret{j}(numb,:);
    numb = T*0.05;
    VaR951{j} = VaR_ret{j}(numb,:);
    numb = T*0.99;
    VaR11{j} = VaR_ret{j}(numb,:);
    numb = T*0.95;
    VaR51{j} = VaR_ret{j}(numb,:);
end

% Monte Carlos VaR's must be added
weight=1/k;

MC_Port_VaR_991_vine=zeros(t,1);
MC_Port_VaR_951_vine=zeros(t,1);
MC_Port_VaR_51_vine=zeros(t,1);
MC_Port_VaR_11_vine=zeros(t,1);

% add specific VaR on any given day 
for i=1:t
    MC_Port_VaR_991_vine(i) = sum(weight.*VaR991{i});
    MC_Port_VaR_951_vine(i) = sum(weight.*VaR951{i});
    MC_Port_VaR_11_vine(i) = sum(weight.*VaR11{i});
    MC_Port_VaR_51_vine(i) = sum(weight.*VaR51{i});
end

% calculate true PuL
% take dataloss due to AR-estimation into account:daten_new
true_PuL = zeros(size(U_pred,1),1);
for i=1:size(U_pred,1)
    true_PuL(i,:) = sum(weight*daten_new(i,:));
end

% VaR (95/1)
figure(15)
a=area(MC_Port_VaR_951_vine(:,1));
set(a,'FaceColor',[1 1 0])
hold on
b=area(MC_Port_VaR_51_vine(:,1));
set(b,'FaceColor',[1 1 0])
hold on
plot(true_PuL,'.')
title('Copula-VaR (95/1) 0.5*HFRXEMN + 0.5*DJW1')

% VaR (99/1)
figure(16)
a=area(MC_Port_VaR_991_vine(:,1));
set(a,'FaceColor',[1 1 0])
hold on
b=area(MC_Port_VaR_11_vine(:,1));
set(b,'FaceColor',[1 1 0])
hold on
plot(true_PuL,'.')
title('Copula-VaR')
title('Copula-VaR (99/1) 0.5*HFRXEMN + 0.5*DJW1')

% Zähel die VaR-Verletzungen (nur left tail), um die Modellgüte zu validieren
for i = 1:t
    if MC_Port_VaR_991_vine(i) > true_PuL(i)
        VaR_Verletzung_991_vine(i) =  1;
    else VaR_Verletzung_991_vine(i) =  0;
    end
end
Summe_Verletzung_VaR_991_vine = sum(VaR_Verletzung_991_vine);
opt_Verletzung = round((t-1)/100);
% Bei 1 hätte man das optimale VaR-Modell
VaR_Verletzung_991_proz_adcc = Summe_Verletzung_VaR_991_vine./opt_Verletzung;

for i = 1:t
    if MC_Port_VaR_951_vine(i) > true_PuL(i)
        VaR_Verletzung_951_vine(i) =  1;
    else VaR_Verletzung_951_vine(i) =  0;
    end
end
Summe_Verletzung_VaR_951_vine = sum(VaR_Verletzung_951_vine);
opt_Verletzung = round((t-1)/100)*5;
VaR_Verletzung_951_proz_vine = Summe_Verletzung_VaR_951_vine./opt_Verletzung;


% Tests nach Christoffersen ("Backtesting"), S.4,
%
% Simple means (unconditional coverage) test H0 : E(It) = p (im Mittel treten p-Verletzungen auf)
% Formel: MT sqrt(T)*(pi - p)/sqrt(VaR(I_t) ist Normalverteilt (0,1)
% I_t: Sequenz von Nullen und Einsen - die hit sequence (Eins: der Verlust der PuL ist größer
% als der VaR)
% pi : sample mean der hit-sequence
% Bilde den Sample Average der hit-sequence
T1 = Summe_Verletzung_VaR_991_vine;
T = size(MC_Port_VaR_991_vine,1);
mean_hit = T1/T;
p = 0.01; %Konfidenzniveau
var_It = var(MC_Port_VaR_991_vine)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.99,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
Vergleich_test1 = [MT norm_krit];
if abs(MT)>norm_krit
    disp('H0 Test1 abgelehnt');
else
    disp('H0 Test 1 angenommen');
end

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
chi_krit = chi2inv(0.99,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
Vergleich_test2 = [LR chi_krit];

if abs(LR)>chi_krit
    disp('H0 Test2 abgelehnt')
else
    disp('H0 Test2 angenommen')
end




