function [Rt_pred,GARCHOutput_Backtest,dccparameters_backtest] = BacktestMultiGARCH(stdresid,start,daten,GARCHOutput,forecastP,Spec,archP,garchQ,GARCHOutput_Backtest)
%
% Backtest for multivariate GARCH models. Computes several univariate GARCH
% models with time-varying information (information will always be added).
% Computes also 1 period forecasted correlation matrix
%
% USAGE:
%      [Rt_pred,AR_pred,ht_pred,resid,GARCHOutput_add] = ...
%      BacktestMultiGARCH(stdresid,start,daten,GARCHOutput,forecastP,Spec,...
%      archP,garchQ,const,GARCHOutput_add)
%
% INPUTS:
%      stdresid:    t x k matrix of stabdardized residuals from univariate
%                   GARCH
%      start:       starting point for first forecast period; first part of
%                   sample will be used for first model
%      daten:       t x k array of return series
%     GARCHOutput:  Structure from estimation of univariate GARCH
%     forecastP:    # of periods to be forecasted
%
% OUTPUTS:
%     Rt_pred: 1 period forecast of time-varying correlation matrix
%     AR_pred: 1 period forecast for conditional mean
%     ht_pred: 1 period forecast of variance
%     resid  : residuals form AR(p)-regression (not forecasted)


[~,k]=size(daten);

% finde das größte AR-Lag heraus, damit nachher alle Daten auf die gleiche
% Länge gebracht werden können
maxarlag=zeros(k,1);
errortype=zeros(1,k);
garchtype=zeros(1,k);
asymmG=zeros(1,k);
for i=1:k
    maxarlag(i)=GARCHOutput{i}.arlag;
    errortype(i) = GARCHOutput{i}.errortype;
    garchtype(i) = GARCHOutput{i}.garchtype;
    asymmG(i) = GARCHOutput{i}.leverage;
end
maxarlag = max(maxarlag);
daten_new = daten(maxarlag+1:end,:);
alignedObservationCount = size(stdresid,1);
if size(daten_new,1) ~= alignedObservationCount
    error('BacktestMultiGARCH:InconsistentSampleLengths', ...
        ['The AR-aligned return sample and standardized residuals must ' ...
         'contain the same number of observations.']);
end
startAligned = start - maxarlag;
if startAligned < 1 || startAligned >= alignedObservationCount
    error('BacktestMultiGARCH:InvalidForecastStart', ...
        'The forecast start is outside the AR-aligned sample.');
end

% estimate mutlivariate GARCH with backtest sample
uniforecastP=Spec.uniforecastP;
count=1;
count_pred=1;
jjj = 1:size(GARCHOutput_Backtest,2); %Variable zählt die univariaten GARCH-Backtest Modelle mit;
c=1;
countuni=1;
backtestStarts = startAligned:forecastP:alignedObservationCount-forecastP;
blockCount = numel(backtestStarts);
dccparameters_backtest = cell(blockCount, 1);
parameters_add = cell(blockCount, 1);
Rt_pred_new = cell(blockCount, 1);
Qt_pred_new = cell(blockCount, 1);
Rt_pred = zeros(k, k, blockCount * forecastP);
Qt_pred = zeros(k, k, blockCount * forecastP);
% da der Originale-Datensatz verarbeitet wird, müssen die Daten um die
% AR-Laglängen gekürzt werden;
% AR_pred = zeros(size(start+1+maxarlag:t,2),k); %für einperiodige Vorhersagen: die letze Vorhersage ist von t_(end-1) auf t_end), d.h. für den letzten Return gibt es keine Vorhersage
% ht_pred = zeros(size(start+1+maxarlag:t,2),k);
% parameters_add=cell(size(GARCHOutput_Backtest,2),1);
% Schätze das MV-GARCH Modell für Multiple Perioden (univariate GARCH und
% MVGARCH werden jeweils zum gleichen Zeitpunkt geschätzt)
for i=backtestStarts
    if strcmp(Spec.DynamicType,'DCC')==1
        dccparameters_backtest{count} = Est_DCC(stdresid(1:i,:), Spec.dccP, Spec.dccQ);
    elseif strcmp(Spec.DynamicType,'ADCC')==1
        dccparameters_backtest{count} = Est_ADCC(stdresid(1:i,:), Spec.dccP, Spec.dccQ);
    elseif strcmp(Spec.DynamicType,'GDCC')==1
        dccparameters_backtest{count} = Est_GDCC(stdresid(1:i,:), Spec.dccP, Spec.dccQ);
    elseif strcmp(Spec.DynamicType,'AGDCC')==1
        dccparameters_backtest{count} = Est_AGDCC(stdresid(1:i,:), Spec.dccP, Spec.dccQ);
    end
    %     machen einperiodigen Forecast für die univariaten Modelle mit den
    %     vorher geschätzten parameters aus GARCHOutput_Backtest_Gauss; nehme
    %     die Daten i+forecastP zur Schätzung, aber dann nur die forecastP
    %     letzen Daten in die Matrix mit den Vorhersagen
%     [AR_pred_all{count},ht_pred_all{count}, resid_all{count}] = ar_univariate_snoop_forecast1P(daten(1:i+maxarlag+forecastP,:),GARCHOutput_Backtest{count},const); %zäle maxarlag dazu da innerhalb der Funktion selber noch einmal um AR-Laglängen korrigiert wird
%     for j=1:size(daten,2)
%         AR_pred_all{count}{j} = AR_pred_all{count}{j}(ardiff(j)+1:end);
%         ht_pred_all{count}{j} = ht_pred_all{count}{j}(ardiff(j)+1:end);
%         resid_all{count}{j} = resid_all{count}{j}(ardiff(j)+1:end);
%         %         Füge die cell-Matrizen zu einer Zeitreihe zusammen;
%         AR_pred(count_pred:count_pred+forecastP-1,j) = AR_pred_all{count}{j}(end-forecastP+1:end);
%         ht_pred(count_pred:count_pred+forecastP-1,j) = ht_pred_all{count}{j}(end-forecastP+1:end);
%         resid(count_pred:count_pred+forecastP-1,j) = resid_all{count}{j}(end-forecastP+1:end);
%     end
    %lese die univariaten GARCH-Parameter aus
    if countuni==uniforecastP*jjj(c)+1 % sorge dafür, dass immer die richtigen univariaten Backtest-Parameter genommen werden; forecastP ist auch der Zeitreum für ein univariates Backtest-Garch Modell sobald forecastP erreicht ist, kommt der nächste Backtest Zeitraum
        c=c+1;
    end
    h=1;
    for l=1:k
        parameters_add{count}(h:h+size(GARCHOutput_Backtest{c}{l}.ParamsGARCH,1)-1,1) = GARCHOutput_Backtest{c}{l}.ParamsGARCH;
        h = h + size(GARCHOutput_Backtest{c}{l}.ParamsGARCH,1);
    end
    %         füge die parameter aus den univariaten Backtest-GARCH und den
    %         Backtest MV-GARCH zusammen
    parameters_add{count}=[parameters_add{count}; dccparameters_backtest{count}];
    %     machen einperiodige Vorhersagen mit den MV-GARCH Modellen und den
    %     dazugehörigen univariaten GARCH; mache für jede forecast-periode
    %     genau eine Vorhersage
    for forecastIndex = 1:forecastP
        prefixData = daten_new(1:i+forecastIndex-1, :);
        if strcmp(Spec.DynamicType,'DCC')==1
            [Rt_pred_new{count}(:,:,forecastIndex), ...
                Qt_pred_new{count}(:,:,forecastIndex)] = DCC_snoop( ...
                parameters_add{count}, prefixData, archP, garchQ, ...
                asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ);
        elseif strcmp(Spec.DynamicType,'ADCC')==1
            [Rt_pred_new{count}(:,:,forecastIndex), ...
                Qt_pred_new{count}(:,:,forecastIndex)] = ADCC_snoop( ...
                parameters_add{count}, prefixData, archP, garchQ, ...
                asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ);
        elseif strcmp(Spec.DynamicType,'GDCC')==1
            [Rt_pred_new{count}(:,:,forecastIndex), ...
                Qt_pred_new{count}(:,:,forecastIndex)] = GDCC_snoop( ...
                parameters_add{count}, prefixData, archP, garchQ, ...
                asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ);
        elseif strcmp(Spec.DynamicType,'AGDCC')==1
            [Rt_pred_new{count}(:,:,forecastIndex), ...
                Qt_pred_new{count}(:,:,forecastIndex)] = AGDCC_snoop( ...
                parameters_add{count}, prefixData, archP, garchQ, ...
                asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ);
        end
    end
    countuni = countuni + forecastP;
    Rt_pred(:,:,count_pred:count_pred+forecastP-1) = Rt_pred_new{count};%fasse die unterschiedlichen cell-matrizen in eine Matrix
    Qt_pred(:,:,count_pred:count_pred+forecastP-1) = Qt_pred_new{count};%fasse die unterschiedlichen cell-matrizen in eine Matrix
    count = count+1;
    count_pred = count_pred + forecastP;
end
