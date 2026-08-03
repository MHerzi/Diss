function [Rt_pred_new,copparameters_backtest,rnd_ret2,Output] = BacktestMultiCopula_paretotail(U,stdresid,GARCHOutput,GARCHOutput_Backtest,MeanForecast,VarianceForecast,PrtFig,daten,Spec)

% Backtest for multivariate Copula models. Computes multivariate Copula
% models (backtest) and several  VaR estimates
%
% USAGE:
%      [Rt_pred_new,copparameters_backtest,rnd_ret,Output] =
%      BacktestMultiCopula(U,stdresid,start,GARCHOutput,GARCHOutput_Backtest,MeanForecast,VarianceForecast,PrtFig,daten,Spec)
%
% INPUTS:
%             U:    t x k array of unif(0,1) variables
%
%      stdresid:    t x k matrix of stabdardized residuals from univariate
%                   GARCH
%
%   GARCHOutput:    Structure from estimation of univariate GARCH
%
% GARCHOutput_Backtest: Structure from estimation of univariate GARCH
% backtest
% 
% MeanForecast: Forecast Mean from univariate models
% 
% VarianceForecast: Forecast Variance from univariate models
% 
% 
% daten: t x k array of returns
% 
% Spec: specification structure from setInputs.m
% 
% 
% OUTPUTS:
% Rt_pred_new: 1-period forecasted correlation matrix
% copparameters_backtest: copula parameters from backtest estimation
% rnd_ret: random returns (used to estimate VaR)
% Output: Structure containing several VaR's and other risk variables
%
% Author:  Martin Grziska, 12/31/2010

% -------------------------------------------------------------------------
% estimate mutlivariate GARCH with backtest sample
[t,k] = size(U);

forecastP=Spec.ForecastNumb;
t2 =  forecastP;
uniforecastP=Spec.uniforecastP;
n=Spec.SimNumb;
start=Spec.ForecastStart;

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
    errortype(i) = GARCHOutput{i}.errortype;
    garchtype(i) = GARCHOutput{i}.garchtype;
    asymmG(i) = GARCHOutput{i}.leverage;
    arlag(i) = GARCHOutput{i}.arlag;
end

maxarlag = max(arlag);
for i=1:k
    if GARCHOutput{i}.arlag<maxarlag
        ardiff(i) = maxarlag-GARCHOutput{i}.arlag;
    else ardiff(i) = 0;
    end
end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

epsilon=10^(-3);
count=1;
jjj = 1:size(GARCHOutput_Backtest,2); %Variable zählt die univariaten GARCH-Backtest Modelle mit;
for i=start:forecastP:t-forecastP
    if strcmp(Spec.CopulaType,'Gauss')
        family=cell(1);
        family{1}='gaussian';
        options = optimset('fmincon');
        options = optimset(options,'Display','iter','TolCon',10^-6,'TolFun',10^-4,'TolX',10^-12,'Algorithm','active-set','Hessian','off','FunValCheck','off','MaxFunEvals',10000);
        if strcmp(Spec.DynamicType,'DCC')
            display('----------------------------------------')
            display('%%%%% Estimating the starting values %%%%%')
            display('----------------------------------------')
            [parameters_start]=Est_DCC(stdresid,Spec.dccP,Spec.dccQ);
            theta0 = [ones(1,Spec.dccP)*parameters_start(end-1)./Spec.dccP; ones(1,Spec.dccQ)*parameters_start(end)/Spec.dccQ];
            lower = zeros(Spec.dccP+Spec.dccQ,1) + 1e-6;
            upper = ones(Spec.dccP+Spec.dccQ,1) - 1e-6;
            display('----------------------------------------')
            display('%%%%% Estimating the DCC Gauss-Copula %%%%%%')
            display('----------------------------------------')
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        elseif strcmp(Spec.DynamicType,'ADCC')
            display('----------------------------------------')
            display('%%%%% Estimating the starting values %%%%%')
            display('----------------------------------------')
            [parameters_start] = Est_ADCC(stdresid, Spec.dccP, Spec.dccQ);
            lower = zeros(Spec.dccP+Spec.dccQ+Spec.dccG,1) + 1e-4;
            upper = ones(Spec.dccP+Spec.dccQ+Spec.dccG,1) - 1e-4;
            display('----------------------------------------')
            display('%%%%% Estimating the ADCC Gauss-Copula %%%%%%')
            display('----------------------------------------')
            theta0 = [(ones(1,Spec.dccP)*parameters_start(end-2)/Spec.dccP); abs(ones(1,Spec.dccP)*parameters_start(end-1)); ones(1,Spec.dccQ)*parameters_start(end)/Spec.dccQ];
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        elseif strcmp(Spec.DynamicType,'GDCC')
            display('----------------------------------------')
            display('%%%%% Estimating the starting values %%%%%')
            display('----------------------------------------')
            parameters_start = Est_DCC(stdresid, Spec.dccP, Spec.dccQ);
            theta0 =  [(ones(k,Spec.dccP)*parameters_start(end-1)/Spec.dccP); ones(k,Spec.dccQ)*parameters_start(end)/Spec.dccQ];
            lower = zeros(k*2,1)+2*options.TolCon;
            upper = ones(k*2,1)-2*options.TolCon;
            display('----------------------------------------')
            display('%%%%% Estimating the GDCC Gauss-Copula %%%%%%')
            display('----------------------------------------')
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        elseif strcmp(Spec.DynamicType,'AGDCC')
            display('----------------------------------------')
            display('%%%%% Estimating the starting values %%%%%')
            display('----------------------------------------')
            [parameters_start] = Est_ADCC(stdresid, Spec.dccP, Spec.dccQ);
            theta0 =  [(ones(k,Spec.dccP)*parameters_start(end-2)/Spec.dccP);  ones(k,Spec.dccP)*parameters_start(end-1); ones(k,Spec.dccQ)*parameters_start(end)/Spec.dccQ];                display('----------------------------------------')
            lower = zeros(k*3,1)+2*options.TolCon;
            upper = ones(k*3,1)-2*options.TolCon;
            display('----------------------------------------')
            display('%%%%% Estimating the AGDCC Gauss-Copula %%%%%%')
            display('----------------------------------------')
            %             [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        end
    elseif strcmp(Spec.CopulaType,'t')
        family=cell(1);
        family{1}='t';
        options = optimset('fmincon');
        options = optimset(options,'Display','iter','TolCon',10^-6,'TolFun',10^-4,'TolX',10^-6,'Algorithm','interior-point','Hessian','off','FunValCheck','off','MaxFunEvals',10000);
        display('----------------------------------------')
        display('%%%%% Estimating the static t-copula %%%%%')
        display('----------------------------------------')
        [kappa_static_t,nuhat_static_t,nuci_static_t] = copulafit('t',U,'Options',options);
        if strcmp(Spec.DynamicType,'DCC')
            [parameters_start]=Est_DCC(stdresid,Spec.dccP,Spec.dccQ);
            theta0 = [ones(1,Spec.dccP)*parameters_start(end-1)./Spec.dccP; ones(1,Spec.dccQ)*parameters_start(end)/Spec.dccQ; nuhat_static_t];
            lower = [zeros(1,2) + 2*options.TolCon 2.1];
            upper = [ones(1,2) - 2*options.TolCon 200];
            display('----------------------------------------')
            display('%%%%% Estimating the DCC-tCopula %%%%%%')
            display('----------------------------------------')
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        elseif strcmp(Spec.DynamicType,'ADCC')
            display('----------------------------------------')
            display('%%%%% Estimating the starting values %%%%%')
            display('----------------------------------------')
            [parameters_start] = Est_ADCC(stdresid, Spec.dccP, Spec.dccQ);
            parameters_start=abs(parameters_start);
            lower = [zeros(1,Spec.dccP+Spec.dccQ+Spec.dccG) + 2*options.TolCon 2.1];
            upper = [ones(1,Spec.dccP+Spec.dccQ+Spec.dccG) - 2*options.TolCon 200];
            theta0 = [parameters_start(end-2)/Spec.dccP; parameters_start(end-1)/Spec.dccP; parameters_start(end)/Spec.dccQ; nuhat_static_t];
            display('----------------------------------------')
            display('%%%%% Estimating the ADCC-tCopula %%%%%%')
            display('----------------------------------------')
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        elseif strcmp(Spec.DynamicType,'GDCC')
            display('----------------------------------------')
            display('%%%%% Estimating the starting values %%%%%')
            display('----------------------------------------')
            parameters_start = Est_DCC(stdresid, Spec.dccP, Spec.dccQ);
            theta0 =  [(ones(k,Spec.dccP)*parameters_start(end-1)/Spec.dccP); ones(k,Spec.dccQ)*parameters_start(end)/Spec.dccQ; 20];
            lower = [zeros(1,k*2)+2*options.TolCon 2.1];
            upper = [ones(1,k*2)-2*options.TolCon 200];
            display('----------------------------------------')
            display('%%%%% Estimating the GDCC tCopula %%%%%%')
            display('----------------------------------------')
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        elseif strcmp(Spec.DynamicType,'AGDCC')
            display('----------------------------------------')
            display('%%%%% Estimating the starting values %%%%%')
            display('----------------------------------------')
            [parameters_start] = Est_ADCC(stdresid, Spec.dccP, Spec.dccQ);
            parameters_start = abs(parameters_start);
            lower = [zeros(1,3*k)+1e-5 2.1];
            upper = [ones(1,k*3)-1e-4 200];
            theta0 =  [(ones(k,Spec.dccP)*parameters_start(end-2)/Spec.dccP);  ones(k,Spec.dccP)*parameters_start(end-1); ones(k,Spec.dccQ)*parameters_start(end)/Spec.dccQ; 20];
            display('----------------------------------------')
            display('%%%%% Estimating the AGDCC-tCopula %%%%%%')
            display('----------------------------------------')
            [copparameters_backtest{count} LL] = fmincon('MultiCopulaLikelihood',theta0,[],[],[],[],lower,upper,'MultiCopulaNonlincon',options,U(1:i,:),Spec.dccP,Spec.dccQ,epsilon,Spec);
        end
    end
     h=0; % mache für jede einzelne Periode innerhalb von forecastP eine Vorhersage
     g=1;
    for jj=1:forecastP
        [Rt_pred_new{count}{jj}] = copulamix_tv_paramforecast_2_grm(copparameters_backtest{count}, 1, family, U(1:i+h,:), Spec.dccP, Spec.dccQ, 1, Spec.DynamicType);
        h=h+1;
        if strcmp(Spec.CopulaType,'t')
            Sim = copularnd_grm('t',Rt_pred_new{count}{jj}{1}{1},copparameters_backtest{count}(end),n);
        elseif strcmp(Spec.CopulaType,'Gauss')
            Sim = copularnd_grm('Gaussian',Rt_pred_new{count}{jj}{1},n);
        end
         if jj==uniforecastP*jjj(g)+1 % sorge dafür, dass immer die richtigen univariaten Backtest-Parameter genommen werden; forecastP ist auch der Zeitreum für ein univariates Backtest-Garch Modell sobald forecastP erreicht ist, kommt der nächste Backtest Zeitraum
            g=g+1;
         end
        for ii=1:k
            rnd_ret=zeros(n,k);
            uinv = GARCHOutput_Backtest{g}{ii}.u_par;
            rnd_ret(:,ii) = Meanforecast(jj,ii) +sqrt(varianceForecast(jj,ii))*uinv.icdf(Sim(:,ii));            
            %   nicht sortierte returns für die Nutzenfunktion
            rnd_ret2{1,jj}(:,ii) = rnd_ret(:,ii);
            %         sortierte returns für den VaR
            rnd_ret(:,ii) = sort(rnd_ret(:,ii));
            %             lese VaR aus
            VaR991{ii}(jj,1) = quantile(rnd_ret(:,ii),0.01);
            VaR951{ii}(jj,1) = quantile(rnd_ret(:,ii),0.05);
            VaR901{ii}(jj,1) = quantile(rnd_ret(:,ii),0.1);
            VaR9991{ii}(jj,1) = quantile(rnd_ret(:,ii),0.001);
            VaR11{ii}(jj,1)  = quantile(rnd_ret(:,ii),0.99);
            VaR51{ii}(jj,1)  = quantile(rnd_ret(:,ii),0.95);
            VaR101{ii}(jj,1)  = quantile(rnd_ret(:,ii),0.9);
            VaR0011{ii}(jj,1)  = quantile(rnd_ret(:,ii),0.999);
            %             rnd_ret{ii}(:,jj) = zeros(n,1); % schreibe zeros
            %             in entsprechende Spalte, damit memory nicht überschritten wird
        end
        clear Sim
        clear rnd_ret
    end
end

% 
% % wenn die copula-parameter mehrmals geschätztz werden, füge die forecasts
% % der einzelnen Perioden zusammen (!!! Achtung die VaR's und rnd_ret2 müssen dann
% folgendermaßen aussehen rnd_ret2{count}{1,jj}(:,ii) ,VaR991{count}{ii}(jj,1)
% if count >1
% %     die ersten  VaR's werden von Hand eingetragen
% for i=1:k
%     VaR991_test{i}=VaR991{1}{i};
% end
% % alle anderen VaR's werden an die ersten angefügt
%     for i=2:count-1
%         for j=1:k
%         VaR991_test{j}(forecastP*(i-1)+1:forecastP*i,1) = VaR991{i}{j};
%         end
%     end
% end


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
Output.PuL = true_PuL;

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
