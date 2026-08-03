function [Rt_pred,copparameters_backtest,U_rnd] = BacktestMultiCopula2(U,stdresid,start,daten,GARCHOutput,forecastP,Spec,archP,garchQ,const,dccP,dccQ,GARCHOutput_Backtest,n)

% Backtest for multivariate GARCH models. Computes several univariate GARCH
% models with time-varying information (information will always be added).
% Computes also 1 period forecasted correlation matrix
%
% USAGE:
%      [Rt_pred,AR_pred,ht_pred,resid] =
%      BacktestMultiCopula(U,stdresid,start,daten,GARCHOutput,forecastP,Spec,archP,garchQ,const,dccP,dccQ)
%
% INPUTS:

%             U:    t x k array of unif(0,1) variables
%
%      stdresid:    t x k matrix of stabdardized residuals from univariate
%                   GARCH
%
%         start:    starting point for first forecast period; first part of
%                   sample will be used for first model
%
%         daten:    t x k array of return series
%
%   GARCHOutput:    Structure from estimation of univariate GARCH
%
%     forecastP:    # of periods to be forecasted
%
%          Spec:   structure containing info about model to be estimated
%
%          dccP:  # of ARCH lags in Dcc specification
%
%          dccG:  # of GARCH lags in Dcc specification
%
% OUTPUTS:
%     Rt_pred: 1 period forecast of time-varying correlation matrix
%     AR_pred: 1 period forecast for conditional mean
%     ht_pred: 1 period forecast of variance
%     resid  : residuals form AR(p)-regression (not forecasted)
%
% Author:  Martin Grziska, 05/04/2010

% -------------------------------------------------------------------------
% estimate mutlivariate GARCH with backtest sample
[t,k] = size(daten);

errortype=zeros(k,1);
garchtype=zeros(k,1);
asymmG=zeros(k,1);
arlag=zeros(k,1);
ardiff=zeros(k,1);
maxarlag=zeros(k,1);
for i=1:k
    errortype(i) = GARCHOutput{i}.errortype;
    garchtype(i) = GARCHOutput{i}.garchtype;
    asymmG(i) = GARCHOutput{i}.leverage;
    maxarlag(i)=GARCHOutput{i}.arlag;
    arlag(i) = GARCHOutput{i}.arlag;
end
maxarlag=max(maxarlag);
for i=1:k
    if GARCHOutput{i}.arlag<maxarlag
        ardiff(i) = maxarlag-GARCHOutput{i}.arlag;
    else ardiff(i) = 0;
    end
    daten_new(:,i) = daten(maxarlag+1:end,i);
end

epsilon=10^(-7);
count=1;
count_pred=1;
indexRt=1;
for i=start:forecastP:t-forecastP
    if strcmp(Spec.CopulaType,'Gauss')
        family=cell(1);
        family{1}='gaussian';
        if strcmp(Spec.DynamicType,'DCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'DCC', 'aus', 'on');
        elseif strcmp(Spec.DynamicType,'ADCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'ADCC', 'aus', 'on');
        elseif strcmp(Spec.DynamicType,'GDCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'GDCC', 'aus', 'on');
        elseif strcmp(Spec.DynamicType,'AGDCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'AGDCC', 'aus', 'on');
        end
    elseif strcmp(Spec.CopulaType,'t')
        family=cell(1);
        family{1}='t';
        if strcmp(Spec.DynamicType,'DCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'DCC', 'aus', 'off'); %schalte Standardfehler und grafiken aus
        elseif strcmp(Spec.DynamicType,'ADCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'ADCC', 'aus', 'off');
        elseif strcmp(Spec.DynamicType,'GDCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'GDCC', 'aus', 'off');
        elseif strcmp(Spec.DynamicType,'AGDCC')
            [Holder, Holder, copparameters_backtest{count}] = copulafitmix_tv_grm2(family, U(1:i,:), 'GDCC', 'aus', 'off');
        end
    end
    %     machen einperiodigen Forecast für die univariaten Modelle mit den
    %     vorher geschätzten parameters aus GARCHOutput_Backtest_Gauss
%     [AR_pred_all{count},ht_pred_all{count}, resid_all{count}] = ar_univariate_snoop_forecast1P(daten(1:i+maxarlag+forecastP,:),GARCHOutput_Backtest{count},const); %zähle maxarlag dazu, da innerhalb der Funktion noch einmal um die AR-laglängen korrigiert wird
    %     [AR_pred_all{count},ht_pred_all{count}, resid_all{count}] = ar_univariate_snoop_forecast1P_new(GARCHOutput_Backtestd{count},const);
%     for j=1:size(daten,2)
%         AR_pred_all{count}{j} = AR_pred_all{count}{j}(ardiff(j)+1:end);
%         ht_pred_all{count}{j} = ht_pred_all{count}{j}(ardiff(j)+1:end);
%         resid_all{count}{j} = resid_all{count}{j}(ardiff(j)+1:end);
%         %         Füge die cell-Matrizen zu einer Zeitreihe zusammen;
%         AR_pred(count_pred:count_pred+forecastP-1,j) = AR_pred_all{count}{j}(end-forecastP+1:end);
%         ht_pred(count_pred:count_pred+forecastP-1,j) = ht_pred_all{count}{j}(end-forecastP+1:end);
%         resid(count_pred:count_pred+forecastP-1,j) = resid_all{count}{j}(end-forecastP+1:end);
%     end
    %         -----------------------------------------------------------------
    %         -----------------------------------------------------------------
    %         alter Code: fügt univariate und copparameter zusammen; ist aber
    %         falsch sowohl für Likelhihood als auch für Copula
    % %
    %         h=1;
    %         for l=1:k
    %             parameters_add{count}(h:h+size(GARCHOutput_Backtest{count}{l}.ParamsGARCH,1)-1,1) = GARCHOutput_Backtest{count}{l}.ParamsGARCH;
    %             h = h + size(GARCHOutput_Backtest{count}{l}.ParamsGARCH,1);
    %         end
    %         %         füge die parameter aus den univariaten Backtest-GARCH und den
    %         %         Backtest MV-GARCH zusammen
    %     end
    %     parameters_add{count}=[parameters_add{count}; copparameters_backtest{count}];
    %     %     machen einperiodige Vorhersagen mit den MV-GARCH Modellen und den
    %     %     dazugehörigen univariaten GARCH
    %     %     [Rt_pred_new{count},Qt_pred_new{count}] = SnoopCopula(parameters_add{count}, resid, archP, garchQ, asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ, GARCHOutput, Spec);
    %     %     Rt_pred(:,:,count_pred:count_pred+forecastP-1) = Rt_pred_new{count}(:,:,end-forecastP+1:end);
    %     %     Qt_pred(:,:,count_pred:count_pred+forecastP-1) = Qt_pred_new{count}(:,:,end-forecastP+1:end);
    
    h=0; % mache für jede einzelne Periode innerhalb von forecastP eine Vorhersage
    for g=1:forecastP
        [Rt_pred_new{count}{g}] = copulamix_tv_paramforecast_2_grm(copparameters_backtest{count}, 1, family, U(1:i+h,:), archP, garchQ, 1, Spec.DynamicType);
        h=h+1;
    end
    for jj=1:forecastP
        Rt_pred(:,:,indexRt) = Rt_pred_new{count}{jj}{1}; % füge die unterschiedlichen Rt_pred-Matrizen in eine einzige
        indexRt=indexRt+1;
    end
    % Erzeuge Zufallszahlen unif(0,1) der jeweiligen Copula für den VaR
    if strcmp(Spec.CopulaType,'t')
        family='t';
        for d=1:forecastP
            U_rnd1{count}{d} = copularnd_grm('t',Rt_pred_new{count}{d}{1},copparameters_backtest{count}(end),n);
        end
    elseif strcmp(Spec.CopulaType,'Gauss')
        for d=1:forecastP
            U_rnd1{count}{d} = copularnd_grm('Gaussian',Rt_pred_new{count}{d}{1},n);
        end
    end
    count = count+1;
    count_pred = count_pred + forecastP;
end
% bringe U_rnd in die Form: jeder Tag steht in einer Zelle, in jeder Zelle
% stehen n-Simulationen für k-Zeitreihen
U_rnd=U_rnd1{1}';
for i=2:count-1
    U_rnd=[U_rnd;U_rnd1{i}'];
end

%
