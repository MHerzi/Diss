function [AR_pred,ht_pred,resid_t] = ar_univariate_snoop_forecast1P(data,GARCH,const)

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
%          data: t x k array of return series (complete series)
%          const: 1 if AR-GARCH is estimated with const, 0 else
%
% OUTPUTS:
%          ht_pred:    predicted vola
%          AR_pred:    predicted residuals
%          resid_t: residuals - data-AR_pred
%          PuL: Profit & Loss
%
% Author: Martin Grziska, 05/20/2010

[t,k] = size(data);

% -----------------------------------------------------------------------
% AR-process 1 period forecast
% -----------------------------------------------------------------------
forecastdata_AR = cell(k,1);
forecastdata2_AR = cell(k,1);
AR_pred = cell(k,1);

% y_t+1 = omega + phi*y_t: lagge die Daten ein lag weniger, als bei den
% univariaten GARCH; der Return in t wird mit dem forecast von t-1 auf t
% verglichen
for i=1:k
    if GARCH{i}.arlag>1
        forecastdata_AR{i}(:,1) = data(:,i);
        %         subtract one from arlag because of forecast!
        forecastdata_AR{i}(:,2:GARCH{i}.arlag) = mlag(data(:,i),GARCH{i}.arlag-1);
    else
        %         must not lag if only one ARlag - needed for forecasting purpose
        forecastdata_AR{i} = data(:,i);
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

% regress estimated coefficients: 1 Period forecast AR; in cell t is
% AR-1Pforecast t+1, e.g.: y100 = omega+y99;
for i=1:k
    AR_pred{i} = (GARCH{i}.ParamsAR'*forecastdata2_AR{i}')';
    % correct data for losses du to arlag
    AR_pred{i} = AR_pred{i}(GARCH{i}.arlag+1:end);
end


% ------------------------------------------------------------------------
% residuals for time t (used in variance-forecast)
% ------------------------------------------------------------------------
residdata=cell(k,1);
for i=1:k
    residdata{i} = mlag(data(:,i),GARCH{i}.arlag);
    if const==1
        residdata{i} = [ones(size(residdata{i},1),1) residdata{i}];
    end
end

% again: correct for observations lost due to AR estimation
resid_t=cell(k,1);
for i=1:k
    resid_t{i} = data(GARCH{i}.arlag+1:end,i)-(GARCH{i}.ParamsAR'*residdata{i}(GARCH{i}.arlag+1:end,:)')';
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
% use last estimated variance for first forecast
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

