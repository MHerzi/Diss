function [AR_pred,ht_pred,resid_t,PuL] = ar_univariate_snoop_forecast1P_new(GARCH,const)

% Data-snooping "forecast": split whole sample in estimation and forecast
% period; use AR and GARCH-parameters from estimation period to predict
% values in forecasting period; forecasting horizon 1; e.g. estimate
% coefficients with first sample, make 1 period forecast for every period
% with fixed coefficients from first period
%
% USAGE:   [AR_pred,ht_pred,resid_t] =...
% ar_univariate_snoop_forecast1P(data,GARCH,const)
%
% INPUTS:
%          GARCH: structure containing parameters etc from univariate
%                 GARCH estimated by AR_MArignalModel.m; sample already
%                 splitted: coefficients already estimated by sample
%                 determined for estimation
%          const: 1 if AR-GARCH is estimated with const, 0 else
%
% OUTPUTS:
%          ht_pred:    predicted vola
%          AR_pred:    predicted residuals
%          resid_t: residuals - data-AR_pred
%          PuL: Profit & Loss
%
% Author: Martin Grziska, 09/03/2010


% -----------------------------------------------------------------------
% AR-process 1 period forecast
% -----------------------------------------------------------------------
k=size(GARCH,1);
forecastdata_AR = cell(k,1);
forecastdata2_AR = cell(k,1);
AR_pred = cell(k,1);
% lag data corresponding to arlags, account for 1 period forecast;
% y_t+1 = omega + phi*y_t
for i=1:k
    if GARCH{i}.arlag>1
        forecastdata_AR{i}(:,1) = GARCH{i}.data(GARCH{i}.arlag+1:end); % daten verlieren durch AR-Schätzung Beobachtungen
        %         subtract one from arlag because of forecast!
        forecastdata_AR{i}(:,2:GARCH{i}.arlag) = mlag(GARCH{i}.data(GARCH{i}.arlag+1:end),GARCH{i}.arlag-1);
    else
        %         must not lag if only one ARlag - needed for forecasting purpose
        forecastdata_AR{i} = GARCH{i}.data(GARCH{i}.arlag+1:end);
    end
end
% add column of ones if const
if const == 1
    for i=1:k
        forecastdata2_AR{i} = [ones(size(forecastdata_AR{i},1),1) forecastdata_AR{i}];
    end
else
    forecastdata2_AR = forecastdata_AR;
end

% for one-period forecast: y_t+1 = omega + phi*yt; e.g. returnseries: 100 x
% 1 matrix last forecast element: y_100 = oemga + phi*y_99, forecast from
% variable y_99 for y_100, so forecast for y_100 can be compared to return
% y_100
for i=1:k
    forecastdata2_AR{i} = mlag(forecastdata2_AR{i},1);
end

% regress estimated coefficients: 1 Period forecast AR; in cell t is
% AR-1Pforecast t+1, e.g.: y100 = omega+y99;
for i=1:k
    AR_pred{i} = (GARCH{i}.ParamsAR'*forecastdata2_AR{i}')';
    % correct data for losses du to arlag
    AR_pred{i} = AR_pred{i}(GARCH{i}.arlag+1:end);
end


% ------------------------------------------------------------------------
% residuals for time t
% ------------------------------------------------------------------------
residdata=cell(k,1);
for i=1:k
    residdata{i} = mlag(GARCH{i}.data,GARCH{i}.arlag); % original data
    if const==1
        residdata{i} = [ones(size(residdata{i},1),1) residdata{i}];
    end
    residdata{i} = residdata{i}(GARCH{i}.arlag+1:end,:); % data minus AR-lags
end

resid_t=cell(k,1);
for i=1:k
    resid_t{i} = GARCH{i}.data(GARCH{i}.arlag+1:end)-(GARCH{i}.ParamsAR'*residdata{i}')';
end
% ------------------------------------------------------------------------


% ------------------------------------------------------------------------
% GARCH-process
% ------------------------------------------------------------------------
% create new ht_variable which last element can be used in recursive
% estimation of predicted ht's
ht_pred=cell(k,1);
resid_t_neg=cell(k,1);
for i=1:k
    ht_pred{i} = GARCH{i}.ht;
    %     make transformation for variance if necessary (for formulas see
    %     Cappiello et al(2006)
    if strcmp(GARCH{i}.GARCH,'TGARCH')
        ht_pred{i} = sqrt(ht_pred{i});
    elseif strcmp(GARCH{i}.GARCH,'AVGARCH')
        ht_pred{i} = sqrt(ht_pred{i});
    elseif strcmp(GARCH{i}.GARCH,'NGARCH')
        ht_pred{i} = ht_pred{i}.^(2/GARCH{i}.ParamsGARCH(4));
    elseif strcmp(GARCH{i}.GARCH,'APGARCH')
        ht_pred{i} = ht_pred{i}.^(2/GARCH{i}.ParamsGARCH(5));
    end
end

% create variable with only negative residuals
for i=1:k
    resid_t_neg{i} = resid_t{i}.*(resid_t{i}<0);
end
% use recursive routine to predict ht with estimated (fixed) GARCH
% coeffcients; every ponit j is  the forecasted value from j-1 to j

for i=1:k
    for j=size(ht_pred{i},1)+1:size(AR_pred{i},1);
        if strcmp(GARCH{i}.GARCH,'GARCH') == 1
            ht_pred{i}(j,:) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*resid_t{i}(j-1,1).^2 + GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1);
        elseif strcmp(GARCH{i}.GARCH,'EGARCH') == 1
            %             !!! EGARCH parameters = ht = omega + gamma + alpha + beta !!!
            ht_pred{i}(j,:) = exp(GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(3)*abs(resid_t{i}(j-1,1))./sqrt(ht_pred{i}(j-1)) + GARCH{i}.ParamsGARCH(2)*resid_t{i}(j-1,1)./sqrt(ht_pred{i}(j-1)) + GARCH{i}.ParamsGARCH(4)*log(ht_pred{i}(j-1)));
        elseif strcmp(GARCH{i}.GARCH,'TGARCH') == 1
            if resid_t_neg{i}(j-1,1)>0
                resid_t_neg{i}(j-1,1)=0;
            end
            ht_pred{i}(j,:) =    GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(resid_t{i}(j-1,1)) + GARCH{i}.ParamsGARCH(3)*resid_t_neg{i}(j-1,1)+GARCH{i}.ParamsGARCH(4)*ht_pred{i}(j-1);
        elseif strcmp(GARCH{i}.GARCH,'GJRGARCH') == 1
            if resid_t_neg{i}(j-1,1)>0
                resid_t_neg{i}(j-1,1)=0;
            end
            ht_pred{i}(j,:) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*resid_t{i}(j-1,1).^2 + GARCH{i}.ParamsGARCH(3)*resid_t_neg{i}(j-1,1).^2+GARCH{i}.ParamsGARCH(4)*ht_pred{i}(j-1);
        elseif strcmp(GARCH{i}.GARCH,'AVGARCH') == 1
            ht_pred{i}(j,:) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(resid_t{i}(j-1,1)) + GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1);
        elseif strcmp(GARCH{i}.GARCH,'NGARCH') == 1
%             ht_pred{i}(j,:)= ((GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(resid_t{i}(j-1,1))^(GARCH{i}.ParamsGARCH(4)) +  GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1)^GARCH{i}.ParamsGARCH(4))^(1/GARCH{i}.ParamsGARCH(4)))^2;
            ht_pred{i}(j,:)= ((GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(resid_t{i}(j-1,1))^(GARCH{i}.ParamsGARCH(4)) +  GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1)^GARCH{i}.ParamsGARCH(4)));
        elseif strcmp(GARCH{i}.GARCH,'NAGARCH') == 1
            ht_pred{i}(j,:) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*(resid_t{i}(j-1,1)+GARCH{i}.ParamsGARCH(4)*sqrt(ht_pred{i}(j-1)))^2 + GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1);
        elseif strcmp(GARCH{i}.GARCH,'APGARCH') == 1
            if resid_t_neg{i}(j-1,1)>0
                resid_t_neg{i}(j-1,1)=0;
            end
            ht_pred{i}(j,:) = (GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(resid_t{i}(j-1,1))^GARCH{i}.ParamsGARCH(4) + GARCH{i}.ParamsGARCH(3)*abs(resid_t_neg{i}(j-1,1))^GARCH{i}.ParamsGARCH(4) + GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1));
        end
    end
end

% make transformations to ensure output is VARIANCE and NOT standard
% deviation
for i=1:k
    if strcmp(GARCH{i}.GARCH,'TGARCH') || strcmp(GARCH{i}.GARCH,'AVGARCH')
        ht_pred{i} = ht_pred{i}.^2;
    elseif strcmp(GARCH{i}.GARCH,'NGARCH')
%         ht_pred{i}=ht_pred{i}.^(2/(GARCH{i}.ParamsGARCH(4)));
    elseif  strcmp(GARCH{i}.GARCH,'APGARCH')
        ht_pred{i}=ht_pred{i}.^(2/(GARCH{i}.ParamsGARCH(5)));
    end
end


% Bestimme größte AR-laglänge, um alle Zeitreihen auf die gleiche
% Datenanzahl zu bringen
maxarlag=zeros(k,1);
arlag=zeros(k,1);
ardiff=zeros(k,1);
for i=1:k
    maxarlag(i)=GARCH{i}.arlag;
    arlag(i) = GARCH{i}.arlag;
end
maxarlag = max(maxarlag);
for i=1:k
    if GARCH{i}.arlag<maxarlag
        ardiff(i) = maxarlag-GARCH{i}.arlag;
    else ardiff(i) = 0;
    end
end

PuL = 0;
for i=1:k
    PuL = PuL+(1/k)*exp(GARCH{i}.data(maxarlag+1:end));
end
PuL = log(PuL);

