function [Rt_pred,VineOutput,CopulaSpec,rnd_ret2,Output] = BacktestVine_multistep2(data,start,CopulaSpec,GARCHOutput_ordered,GARCHOutput_Backtest_ordered,MeanForecast,VarianceForecast,PrtFig,daten,Spec)
                                                                          

% Function that estimates VaR and Christoffersen's HIT-ratios for
% CopulaVine Functions; univariate GARH models are estimated with ever
% incrreasing information
%
% USAGE:
%         [AR_pred,ht_pred,resid,Rt_pred,VineOutput,CopulaSpec] =...
%         BacktestVine(data,GARCHOutput,start,forecastP)
%
% INPUT:
%           data:      t x k array of unif(0,1) (already corrected for
%           AR-lags)
%    GARCHOutput:      Structure from univariate GARCH estimation
%          start:      period +1 forecasting starts with
%      forecastP:      # of periods for one period ahead forecasts
%
% OUTPUT:
%         Rt_pred:     1-period forecasted time-varying correlation matrix
%      VineOutput:     Structure containing information about estimated vine
%      CopulaSpec:     Structure from vine estimation
%
% Author: Martin Grziska
% last modification: 09/10/2010

% ------------------------------------------------------------------------

[t,k] = size(data);
forecastP=Spec.ForecastNumb;
t2 =  size(MeanForecast,1);
uniforecastP=Spec.uniforecastP;
n=Spec.SimNumb;

% rnd_ret = cell(1,k);
rnd_ret2 = cell(1,k);
% for i=1:k
%     rnd_ret{1,i} = zeros(n,t);
% end
VaR991 = cell(1,k);
for i=1:k
    VaR991{1,i} = zeros(t2,1);
end
VaR951 = cell(1,k);
for i=1:k
    VaR951{1,i} = zeros(t2,1);
end
VaR901 = cell(1,k);
for i=1:k
    VaR901{1,i} = zeros(t2,1);
end
VaR9991 = cell(1,k);
for i=1:k
    VaR9991{1,i} = zeros(t2,1);
end
VaR11 = cell(1,k);
for i=1:k
    VaR11{1,i} = zeros(t2,1);
end
VaR51 = cell(1,k);
for i=1:k
    VaR51{1,i} = zeros(t2,1);
end
VaR101 = cell(1,k);
for i=1:k
    VaR101{1,i} = zeros(t2,1);
end
VaR0011 = cell(1,k);
for i=1:k
    VaR0011{1,i} = zeros(t2,1);
end

errortype=zeros(k,1);
garchtype=zeros(k,1);
asymmG=zeros(k,1);
arlag=zeros(k,1);
ardiff=zeros(k,1);

for i=1:k
    errortype(i) = GARCHOutput_ordered{i}.errortype;
    garchtype(i) = GARCHOutput_ordered{i}.garchtype;
    asymmG(i) = GARCHOutput_ordered{i}.leverage;
    arlag(i) = GARCHOutput_ordered{i}.arlag;
end

maxarlag = max(arlag);
for i=1:k
    if GARCHOutput_ordered{i}.arlag<maxarlag
        ardiff(i) = maxarlag-GARCHOutput_ordered{i}.arlag;
    else ardiff(i) = 0;
    end
end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

% estimate vine copula for backtest purposes
count=1;
countret=1;
jjj = 1:size(GARCHOutput_Backtest_ordered,2); %Variable zählt die univariaten GARCH-Backtest Modelle mit;
g=1;
for i=start:forecastP:t-forecastP %Schätze Koeffizienten bis t-forecastP so dass von t-forecastP+1 bis t Vorhersagen gemacht werden
    [Holder, VineOutput{count}] = fitCopulaVine_multistep_grm(data(1:i,:),CopulaSpec);
    h=0;
    for jj = 1:forecastP
        [Rt_pred{count}{jj}] = CopulaVineLL_grm_1P_forecast(VineOutput{count}.VineParams,data(1:i+h,:),CopulaSpec); % es werden einperiodige Vorhersagen gemacht i+forecastP Vorhersagen gemacht
        h=h+1;
        Sim = SimVine(n,CopulaSpec,VineOutput{count},Rt_pred{count}{jj}); %simuliere für jeden Zeitpunk t n-Variablen für den MC-VaR
        if countret==uniforecastP*jjj(g)+1 % sorge dafür, dass immer die richtigen univariaten Backtest-Parameter genommen werden; forecastP ist auch der Zeitreum für ein univariates Backtest-Garch Modell sobald forecastP erreicht ist, kommt der nächste Backtest Zeitraum
            g=g+1;
        end
        for ii=1:k
            rnd_ret=zeros(n,k);
            if strcmp(GARCHOutput_ordered{ii}.dist,'GAUSS') == 1
                %                 rnd_ret{1,ii}(:,jj)= MeanForecast(jj,ii) + sqrt(VarianceForecast(jj,ii))*norminv(Sim{jj}(:,ii));
                rnd_ret(:,ii)= MeanForecast(countret,ii) + sqrt(VarianceForecast(countret,ii))*norminv(Sim(:,ii));
            elseif strcmp(GARCHOutput_ordered{ii}.dist,'STUDENTST') == 1
                rnd_ret(:,ii) = MeanForecast(countret,ii) + sqrt(VarianceForecast(countret,ii))*tinv(Sim(:,ii),GARCHOutput_Backtest_ordered{g}{ii}.ParamsGARCH(end));
            elseif strcmp(GARCHOutput_ordered{ii}.dist,'GED') == 1
                rnd_ret(:,ii) = MeanForecast(countret,ii) + sqrt(VarianceForecast(countret,ii))*gedinv_grm(Sim(:,ii),GARCHOutput_Backtest_ordered{g}{ii}.ParamsGARCH(end));
            elseif strcmp(GARCHOutput_ordered{ii}.dist,'SKEWT') == 1
                rnd_ret(:,ii) = MeanForecast(countret,ii) + sqrt(VarianceForecast(countret,ii))*skewtdis_inv_grm(Sim(:,ii),GARCHOutput_Backtest_ordered{g}{ii}.ParamsGARCH(end-1),GARCHOutput_Backtest_ordered{g}{ii}.ParamsGARCH(end));
            end
            %   nicht sortierte returns für die Nutzenfunktion
            rnd_ret2{1,countret}(:,ii) = rnd_ret(:,ii);
            %         sortierte returns für den VaR
            rnd_ret(:,ii) = sort(rnd_ret(:,ii));
            %             lese VaR aus
            VaR991{ii}(countret,1) = quantile(rnd_ret(:,ii),0.01);
            VaR951{ii}(countret,1) = quantile(rnd_ret(:,ii),0.05);
            VaR901{ii}(countret,1) = quantile(rnd_ret(:,ii),0.1);
            VaR9991{ii}(countret,1) = quantile(rnd_ret(:,ii),0.001);
            VaR11{ii}(countret,1)  = quantile(rnd_ret(:,ii),0.99);
            VaR51{ii}(countret,1)  = quantile(rnd_ret(:,ii),0.95);
            VaR101{ii}(countret,1)  = quantile(rnd_ret(:,ii),0.9);
            VaR0011{ii}(countret,1)  = quantile(rnd_ret(:,ii),0.999);
            %             rnd_ret{ii}(:,jj) = zeros(n,1); % schreibe zeros
            %             in entsprechende Spalte, damit memory nicht überschritten wird
        end
        countret = countret + 1;
        clear Sim
        clear rnd_ret
    end
    count=count+1;
end

% Bilde Portfolio VaR
Port991 = zeros(t2,1);
for i=1:k
    Port991 = Port991+(1/k)*VaR991{i};
end
Output.VaRPortfolio991 = Port991;

% Bilde Portfolio VaR
Port951 = zeros(t2,1);
for i=1:k
    Port951 = Port951+(1/k)*VaR951{i};
end
Output.VaRPortfolio951 = Port951;

Port901 = zeros(t2,1);
for i=1:k
    Port901 = Port901+(1/k)*VaR901{i};
end
Output.VaRPortfolio901 = Port901;

Port9991 = zeros(t2,1);
for i=1:k
    Port9991 = Port9991+(1/k)*VaR9991{i};
end
Output.VaRPortfolio9991 = Port9991;

Port11 = zeros(t2,1);
for i=1:k
    Port11 = Port11+(1/k)*VaR11{i};
end
Output.VaRPortfolio11 = Port11;

Port51 = zeros(t2,1);
for i=1:k
    Port51 = Port51+(1/k)*VaR51{i};
end
Output.VaRPortfolio51 = Port51;

Port101 = zeros(t2,1);
for i=1:k
    Port101 = Port101+(1/k)*VaR101{i};
end
Output.VaRPortfolio101 = Port101;

Port0011 = zeros(t2,1);
for i=1:k
    Port0011 = Port0011+(1/k)*VaR0011{i};
end
Output.VaRPortfolio0011 = Port0011;

% Berechne die wahre PuL
% Es gehen Beobachtungen durch die AR-Schätzung verloren (in diesem Fall
% eine)
true_PuL=zeros(t2,1);
for i=1:k
    true_PuL = true_PuL + 1/k*exp(daten(end-t2+1:end,i)); %für den einperiodigen Forecast muss eine Beobachtung abgezogen werden, da der letze Return nicht mehr zum Forecast benutz werden kann
end
true_PuL = log(true_PuL);

% mache den Backtest nur auf den out-of-sample Ergebnissen
% VaR (95/1)
if strcmp(PrtFig,'on')
    figure
    a=area(Port951(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port51(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,:),'.')
    title('VaR (95/1)')
    % VaR (99/1)
    figure
    a=area(Port991(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port11(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,1),'.')
    title('VaR (99/1)')
    % VaR (90/1)
    figure
    a=area(Port901(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port101(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,:),'.')
    title('VaR (90/1)')
    % VaR (999/1)
    figure
    a=area(Port9991(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port0011(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,:),'.')
    title('VaR (999/1)')
end

% Zähel die VaR-Verletzungen (nur left tail), um die Modellgüte zu validieren
VaRviol991=zeros(size(Port991,1),1);
for i = 1:size(Port991,1)
    if Port991(i) > true_PuL(i)
        VaRviol991(i) =  1;
    else VaRviol991(i) =  0;
    end
end
SumVaRviol991 = sum(VaRviol991);
Output.SumVaRviol991 = sum(VaRviol991);

VaRviol951=zeros(size(Port951,1),1);
for i = 1:size(Port951,1)
    if Port951(i) > true_PuL(i)
        VaRviol951(i) =  1;
    else VaRviol951(i) =  0;
    end
end
SumVaRviol951 = sum(VaRviol951);
Output.SumVaRviol951 = sum(VaRviol951);

VaRviol901=zeros(size(Port901,1),1);
for i = 1:size(Port901,1)
    if Port901(i) > true_PuL(i)
        VaRviol901(i) =  1;
    else VaRviol901(i) =  0;
    end
end
SumVaRviol901 = sum(VaRviol901);
Output.SumVaRviol901 = sum(VaRviol901);

VaRviol9991=zeros(size(Port9991,1),1);
for i =  1:size(Port9991,1)
    if Port9991(i) > true_PuL(i)
        VaRviol9991(i) =  1;
    else VaRviol9991(i) =  0;
    end
end
SumVaRviol9991 = sum(VaRviol9991);
Output.SumVaRviol9991 = sum(VaRviol9991);

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
T = size(Port991,1);
mean_hit = T1/T;
p = 0.01; %Konfidenzniveau (promised probability)
var_It = var(VaRviol991)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 1-normcdf(MT,0,1);
Output.UncondCoverage991_MT = [MT norm_krit pval];

% für VaR95-1
T1 = SumVaRviol951;
T = size(Port951,1);
mean_hit = T1/T;
p = 0.05; %Konfidenzniveau
var_It = var(VaRviol951)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 1-normcdf(MT,0,1);
Output.UncondCoverage951_MT = [MT norm_krit pval];

% für VaR90-1
T1 = SumVaRviol901;
T = size(Port901,1);
mean_hit = T1/T;
p = 0.1; %Konfidenzniveau
var_It = var(VaRviol901)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 1-normcdf(MT,0,1);
Output.UncondCoverage901_MT = [MT norm_krit pval];

% für VaR999-1
T1 = SumVaRviol9991;
T = size(Port9991,1);
mean_hit = T1/T;
p = 0.001; %Konfidenzniveau
var_It = var(VaRviol9991)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 1-normcdf(MT,0,1);
Output.UncondCoverage9991_MT = [MT norm_krit pval];

% -----------------------------------------------------------------------
% Likelihood ratio test
% Nullhypothese pi = p, also erwartete Wert der hit-Sequenz (pi) soll dem
% coverage ratio (p) entsprechen
T1 = SumVaRviol991;
T = size(Port991,1);
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
T1 = SumVaRviol951;
T = size(Port951,1);
T0 = T-T1;
p=0.05;
% optimierte Likelihood
Like1_951 = (1-T1/T)^T0*(T1/T)^T1;
Like2_951 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR951 = -2*log(Like2_951/Like1_951);
% kritischer Wert der Chi^2-Verteilung
krit_chi951 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR951,1);
Output.Independence951_LR = [LR951 krit_chi951 pval];


% für VaR 901
T1 = SumVaRviol901;
T = size(Port901,1);
T0 = T-T1;
p=0.1;
% optimierte Likelihood
Like1_901 = (1-T1/T)^T0*(T1/T)^T1;
Like2_901 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR901 = -2*log(Like2_901/Like1_901);
% kritischer Wert der Chi^2-Verteilung
krit_chi901 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR901,1);
Output.Independence901_LR = [LR901 krit_chi901 pval];

% für VaR 9991
T1 = SumVaRviol9991;
T = size(Port9991,1);
T0 = T-T1;
p=0.1;
% optimierte Likelihood
Like1_9991 = (1-T1/T)^T0*(T1/T)^T1;
Like2_9991 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR9991 = -2*log(Like2_9991/Like1_9991);
% kritischer Wert der Chi^2-Verteilung
krit_chi9991 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR9991,1);
Output.Independence9991_LR = [LR9991 krit_chi9991 pval];

% -----------------------------------------------------------------------
% Ljung-Box (independence testing)
% siehe(Jin (2009), "Large Portfolio Risk Management with Dynamic Copulas)
% H0: model fit is adequate
[H_991,pValue_991,Qstat_991,CriticalValue_991] = lbqtest(VaRviol991,5,.10,5);
Output.LB991 = [Qstat_991  CriticalValue_991 pValue_991];
% für VaR 951
[H_951,pValue_951,Qstat_951,CriticalValue_951] = lbqtest(VaRviol951,5,.10,5);
Output.LB951 = [Qstat_951 CriticalValue_951 pValue_951];
% für VaR 901
[H_901,pValue_901,Qstat_901,CriticalValue_901] = lbqtest(VaRviol901,5,.10,5);
Output.LB901 = [Qstat_901 CriticalValue_901 pValue_901];
% für VaR 9991
[H_9991,pValue_9991,Qstat_9991,CriticalValue_9991] = lbqtest(VaRviol9991,5,.05,5);
Output.LB9991 = [Qstat_9991 CriticalValue_9991 pValue_9991];


% lower partial moment
ELPM991 = cell(k,1);
for j=1:k
    for i=1:size(VaR991{k},1)
        ELPM991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR991{j}(i),[0 1 2 3]);
    end
end
Output.ELPM991 = ELPM991;

ELPM951 = cell(k,1);
for j=1:k
    for i=1:size(VaR951{k},1)
        ELPM991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR951{j}(i),[0 1 2 3]);
    end
end
Output.ELPM951 = ELPM951;

ELPM901 = cell(k,1);
for j=1:k
    for i=1:size(VaR901{k},1)
        ELPM901{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR901{j}(i),[0 1 2 3]);
    end
end
Output.ELPM901 = ELPM901;

ELPM9991 = cell(k,1);
for j=1:k
    for i=1:size(VaR9991{k},1)
        ELPM9991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR9991{j}(i),[0 1 2 3]);
    end
end
Output.ELPM9991 = ELPM9991;
% [VaR,SimReturn] = VineVaR(Sim,GARCHOutput_ordered,GARCHOutput_Backtest_ordered,Spec.uniforecastP,Spec.SimNumb,AR_pred_ordered,ht_pred_ordered,PrtFig,daten);
% clear Sim
