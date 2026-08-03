function [AR_pred,ht_pred,resid,Rt_pred,VineOutput,CopulaSpec] = BacktestVine(data,GARCHOutput,start_new,forecastP,GARCHOutput_add,CopulaSpec,const,daten)

% Function that estimates VaR and Christoffersen's HIT-ratios for
% CopulaVine Functions; univariate GARH models are estimated with ever
% incrreasing information
%
% USAGE:
%         [AR_pred,ht_pred,resid,Rt_pred,VineOutput,CopulaSpec] =...
%         BacktestVine(data,GARCHOutput,start,forecastP)
%
% INPUT:
%           data:      t x k array of unif(0,1) 
%    GARCHOutput:      Structure from univariate GARCH estimation
%          start:      period +1 forecasting starts with 
%      forecastP:      # of periods for one period ahead forecasts
%
% OUTPUT:
%         AR_pred:     1-period forecasted conditional mean
%         ht_pred:     1-period forecasted conditional variance
%           resid:     residuals from AR-regression
%         Rt_pred:     1-period forecasted time-varying correlation matrix
%      VineOutput:     Structure containing information about estimated vine
%      CopulaSpec:     Structure from vine estimation


% ------------------------------------------------------------------------


[t,k]=size(data);
% -----------------------------------------------------------------------
% bring all GARCHOutput vectors to same length
maxarlag = zeros(k,1);
arlag = zeros(k,1);
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
%     daten_new(:,i) = data(maxarlag+1:end,i);
    U_new(:,i) = GARCHOutput{i}.U(ardiff(i)+1:end,:);
end
% ------------------------------------------------------------------------
[t,k]=size(U_new);
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% estimate vine copula for backtest purposes
[LogL, VineOutput, CopulaSpec] = fitCopulaVine_grm(U_new(1:start_new,:),CopulaSpec);

if strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1
    % use estimated coefficients from vine to forcast correlation matrix for
    % one period:
    % transform cell VineOutput.VineParams to vector
    h=1;
    g=size(VineOutput.VineParams,2);
    phi = cell(.5*k*(k-1),1);
    for i=1:size(VineOutput.VineParams,2)
        for c=1:g
            phi{h} = VineOutput.VineParams{i,c};
            h=h+1;
        end
        g=g-1;
    end
    theta=zeros(.5*k*(k-1)*size(phi{1}{1},1),1);
    h=1;
    for i=1:size(phi{1}{1},1):.5*k*(k-1)*size(phi{1}{1},1)
        theta(i:i+size(phi{1}{1},1)-1) = phi{h}{:,:};
        h=h+1;
    end
end
% use parameters-vector to estimate one period forecast correlation matrix
[LogL,Rt_pred]=CopulaVineLL_grm_1P_forecast(theta,U_new,CopulaSpec);

g=1;
l=length(daten);
for i=start_new:forecastP:l-forecastP
    [AR_pred_all{g},ht_pred_all{g}, resid_all{g}] = ar_univariate_snoop_forecast1P(daten(1:i+forecastP,:),GARCHOutput_add{g},const);
    g=g+1;
end

% durch (eventuell) unterschiedliche AR-lags gehen (eventuell)
% Beobachtungen verloren; bringe alle auf gleich Länge
for i=1:g-1
    for j=1:size(daten,2)
        AR_pred_all{i}{j} = AR_pred_all{i}{j}(ardiff(j)+1:end);
        ht_pred_all{i}{j} = ht_pred_all{i}{j}(ardiff(j)+1:end);
        resid_all{i}{j} = resid_all{i}{j}(ardiff(j)+1:end);
    end
end

% Bringe die Daten der jeweiligen Forecast-Periode in eine Zeitreihe - im
% Beispiel steht also von 1:start_new + Länge des forecast (also z.B. 20 Perioden) die Daten, die mit dem ersten Satz
% Koeffizienten geschätzt wurden, danach folgen 20 Daten, die mit dem
% zweiten Satz Koeffizienten geschätzt wurden usw.; beachte die 20 Perioden
% sind 20 1-periodige forecasts
% füge die unterscheidlichen Zeiträume (mit den unterschiedlich geschätzten
% Parametern) in eine Zeitreihe (pro Datenzeitreihe) zusammen

numb = 1:forecastP;
for j=1:size(AR_pred_all,2)
    for h=1:k
        AR_pred(numb(1):numb(end),h) = AR_pred_all{j}{h}(end-forecastP+1:end)';
        ht_pred(numb(1):numb(end),h) = ht_pred_all{j}{h}(end-forecastP+1:end)';
        resid(numb(1):numb(end),h) = resid_all{j}{h}(end-forecastP+1:end)';
    end
    numb=numb+forecastP;
end


