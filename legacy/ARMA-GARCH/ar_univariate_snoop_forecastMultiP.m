function [AR_pred,ht_pred,resid_pred] = ar_univariate_snoop_forecastMultiP(data,GARCH,const)
% Data-snooping "forecast": split whole sample in estimation and forecast
% period; use AR and GARCH-parameters from estimation period to predict
% values in forecasting period; forecasting period 1; e.g. estimate
% coefficients with first sample, make 1 period forecast for every period
% with fixed coefficients from first period
%
% USAGE:   [AR_pred,ht_pred] = ar_univariate_snoop_forecast1P(data,GARCH,const)
%
% INPUTS:
%          GARCH: structure containing parameters etc from univariate
%                 GARCH estimated by AR_MarignalModel.m; sample already
%                 splitted: coefficients already estimated by sample
%                 determined for estimation
%          data: t x k array of return series (complete series)
%          const: 1 if AR-GARCH is estimated with const, 0 else
%
% OUTPUTS:
%          ht_pred:    predicted vola
%          AR_pred:    predicted residuals
%          resid_t: residuals - data-AR_pred
%
% Author: Martin Grziska, 04/10/2010

[t,k] = size(data);

% -----------------------------------------------------------------------
% AR-process 1 period forecast
% -----------------------------------------------------------------------
lagdata_AR=cell(k,1);
lagdata2_AR=cell(k,1);
AR_pred=cell(k,1);
resid_t = cell(k,1);
resid_pred = cell(k,1);
% lag data corresponding to arlags, account for 1 period forecast; 
% y_t+1 = omega + phi'y_t
for i=1:k
    if GARCH{i}.arlag>1
        lagdata_AR{i}(:,1) = data(:,i);
        lagdata_AR{i}(:,2:GARCH{i}.arlag) = mlag(data(:,i),GARCH{i}.arlag-1);
        %     cut for observations lost due to AR-estimation
        lagdata_AR{i} = trimr(lagdata_AR{i},GARCH{i}.arlag-1,0);
    else
        %         must not lag if only one ARlag - needed for forecasting purpose
        lagdata_AR{i} = data(:,i);
    end
end
% add column of ones if const
if const == 1
    for i=1:k
        lagdata2_AR{i} = [ones(size(lagdata_AR{i},1),1) lagdata_AR{i}];
%         cut last observation

    end
else
    lagdata2_AR = lagdata_AR;
end


% regress estimated coefficients: 1 Period forecast AR
for i=1:k
    AR_pred{i} = GARCH{i}.ParamsAR'*lagdata2_AR{i}';
end
% ------------------------------------------------------------------------
% Infer residuals for time t
% ------------------------------------------------------------------------
% trim original data for observations lost due to AR
for i=1:k
    data_new{i}= trimr(data(:,i),GARCH{i}.arlag,0);
end
% residuals must be inferred at time t, so lag data and make regression again
for i=1:k
    resid_t{i} = data_new{i} - trimr((GARCH{i}.ParamsAR'*mlag(lagdata2_AR{i},1)')',1,0);
    resid_pred{i}=resid_t{i};
end
% ------------------------------------------------------------------------


% ------------------------------------------------------------------------
% GARCH-process
% ------------------------------------------------------------------------
% create new ht_variable which last elemtn can be used in recursive
% estimation of predicted ht's
ht_pred=cell(k,1);
for i=1:k
    ht_pred{i} = GARCH{i}.ht;
end
% use recursive routine to predict ht with estimated (fixed) GARCH
% coeffcients; every ponit j is  the forecasted value from j-1 to j
for i=1:k
    for j=size(ht_pred{i},1)+1:size(AR_pred{i},2);
        if strcmp(GARCH{i}.GARCH,'GARCH') == 1
            ht_pred{i}(j,:) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*resid_t{i}(j-1,1).^2 + GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1);
        elseif strcmp(GARCH{i}.GARCH,'EGARCH') == 1
            %             !!! EGARCH parameters = ht = omega + gamma + alpha + beta !!!
            ht_pred{i}(j,:) = exp(GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(3)*abs(resid_t{i}(j-1,1))./sqrt(ht_pred{i}(j-1)) + GARCH{i}.ParamsGARCH(2)*resid_t{i}(j-1,1)./sqrt(ht_pred{i}(j-1)) + GARCH{i}.ParamsGARCH(4)*log(ht_pred{i}(j-1)));
        elseif strcmp(GARCH{i}.GARCH,'TGARCH') == 1
            %             Cappiello et al (2006): TGARCH models sqrt of ht
            ht_pred{i}(j-1) = sqrt(ht_pred{i}(j-1));
            if resid_t{i}(j-1,1)>0
                resid_t{i}(j-1,1)=0;
            end
            ht_pred{i}(j,:) =    GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(resid_t{i}(j-1,1)) + GARCH{i}.ParamsGARCH(3)*resid_t{i}(j-1,1)+GARCH{i}.ParamsGARCH(4)*ht_pred{i}(j-1);
            ht_pred{i}(j,:) = ht_pred{i}(j-1,:).^2;
        elseif strcmp(GARCH{i}.GARCH,'GJRGARCH') == 1
            if resid_t{i}(j-1,1)>0
                resid_t{i}(j-1,1)=0;
            end
            ht_pred{i}(j,:) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*resid_t{i}(j-1,1).^2 + GARCH{i}.ParamsGARCH(3)*resid_t{i}(j-1,1).^2+GARCH{i}.ParamsGARCH(4)*ht_pred{i}(j-1);
        end
    end
end



