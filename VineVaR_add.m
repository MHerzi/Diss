function [VaR,HIT] = VineVaR_add(data,VineOutput,GARCHOutput,CopulaSpec,T)

% Function that estimates VaR and Christoffersen's HIT-ratios for
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
% bring all GARCHOutput vectors to same length
for i=1:k
    maxarlag(i)=GARCHOutput{i}.arlag;
    arlag(i) = GARCHOutput{i}.arlag;
end
maxarlag=max(maxarlag);

for i=1:k
    if GARCHOutput{i}.arlag<maxarlag
        ardiff(i) = maxarlag-GARCHOutput{i}.arlag;
    else ardiff(i) = 0;
    end
    ht_new(:,i) = GARCHOutput{i}.ht(ardiff(i)+1:end);
    resid_new(:,i) = GARCHOutput{i}.Innovations(ardiff(i)+1:end);
    stdresid(:,i) = resid_new(:,i)./sqrt(ht_new(:,i));
    scores_new{i} = GARCHOutput{i}.Scores(ardiff(i)+1:end,:);
    daten_new(:,i) = daten(maxarlag+1:end,i);
end
% ------------------------------------------------------------------------

% estimate GARCH models with timey varying window, where information is
% alway increasing
k=1;
for i=500:forecastP:t
    [GARCHOutput_add{k}] = AR_MarginalModel(daten(1:i,:),archP(1),garchQ(1),const,GARCHOutput);
    k=k+1;
end

% estimate residuals, variances and residuals for every pair of estimated
% coeffcients
k=1;
for i=500:forecastP:t-forecastP
    [AR_pred_all{k},ht_pred_all{k}, resid_pred_all{k}] = ar_univariate_snoop_forecast1P(daten(1:i+forecastP+maxarlag-1,:),GARCHOutput_add{k},const);
    k=k+1;
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
k=size(daten,2);
for i=1:k
    AR_pred(:,i)=AR_pred_all{1}{i}';
    ht_pred(:,i)=ht_pred_all{1}{i};
    resid_pred(:,i)=resid_pred_all{1}{i};
end

g=2;
for j=500+forecastP:forecastP:t-forecastP
    for h=1:k
        AR_pred(j:j+forecastP-1,h) = AR_pred_all{g}{h}(end-forecastP+1:end)';
        ht_pred(j:j+forecastP-1,h) = ht_pred_all{g}{h}(end-forecastP+1:end)';
        resid_pred(j:j+forecastP-1,h) = resid_pred_all{g}{h}(end-forecastP+1:end)';
    end
    g=g+1;
end
% -------------------------------------------------------------------------
%             Simulate unif(0,1) data from vine
corrspec=CopulaSpec.corrspec;
[t,k]=size(data);
Rt=cell(k-1,k-1);
Sim=cell(k-1,k-1);
if strcmp(corrspec,'Patton') == 1 || strcmp(corrspec,'DCC') == 1 || strcmp(corrspec,'TVC') == 1
    for h=1:t
            c=k-1;
        for i=1:.5*(k*(k-1))
            for j=1:c
                Rt{i,j} = VineOutput.Rt{i,j}(2,1,h);
            end
            c=c-1;
        end
        [Sim{h}] = SimVine(T,CopulaSpec,VineOutput,Rt);
    end
end
    

% -----------------------------------------------------------------------
%               VaR-calculation
% ----------------------------------------------------------------------

[t,k] = size(data);
stdresid = cell(t,1);
% create stdresids from U's to generate returns 
% every cell i represents one day
for i=1:t
    for j=1:k
        if strcmp(GARCHOutput{j}.dist,'Gaussian') == 1;
            stdresid{i}(:,j) = norminv(Sim{i}(:,j));
        elseif strcmp(GARCHOutput{j}.dist,'t') == 1;
            stdresid{i}(:,j) = tinv(Sim{i}(:,j),GARCHOutput{j}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput{j}.dist,'GED') == 1;
            stdresid{i}(:,j) = gedinv(Sim{i}(:,j),GARCHOutput{j}.ParamsGARCH(end));
        elseif strcmp(GARCHOutput{j}.dist,'SKEWT') == 1;
            stdresid{i}(:,j) = skewtdis_inv(Sim{i}(:,j),GARCHOutput{j}.ParamsGARCH(end-1),GARCHOutput{j}.ParamsGARCH(end));
        end
    end
end



% Generiere aus den standardisierten Returns wieder nicht-standardisierte
% benutze dazu den forecast für den Mittelwert und die Varianz
% write AR_pred results in one cell
AR_pred_new = cell(1,1);
ht_pred_new = cell(1,1);
for i=1:k
    AR_pred_new{1}(:,i) = AR_pred{i};
    ht_pred_new{1}(:,i) = ht_pred{i};
end
VaR_ret=cell(t,1);
% generate returns for VaR using predicted mean and predicted vola
for j=1:t
    for i=1:T
        VaR_ret{j}(i,:) = AR_pred_new{1}(j,:)+sqrt(ht_pred_new{1}(j,:)).*stdresid{j}(i,:);
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
true_PuL = zeros(t,1);
for i=1:t
    true_PuL(i,:) = sum(weight*data(i,:));
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
    'H0 Test1 abgelehnt'
else
    'H0 Test 1 angenommen'
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
    'H0 Test2 abgelehnt'
else
    'H0 Test 2 angenommen'
end




