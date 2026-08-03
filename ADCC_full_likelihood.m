function [logL, Rt, likelihoods, Qt] = ADCC_full_likelihood(parameters, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ, dccG)
% PURPOSE:
%        Restricted likelihood for use in the ADCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC parameters
%
% USAGE:
%        [logL, Rt, likelihoods, Qt] = ADCC_full_likelihood(params, stdresud, P, Q)
%
% INPUTS:
%    params      - A 2*P+Q by 1 vector of parameters of the form [dccPparameters; dccQparameters, adccPparameters]
%    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
%
%    garchtype:  Model        garchtype-number
%                EGARCH       0
%                GARCH        1
%                TGARCH       2
%                AVGARCH      3
%                NAGRCH       4
%                NAGARCH      5
%                APGARCH      6
%                ALLGARCH     7
%                GJRGARCH     8
%
%    errortype: 'NORMAL':     1
%               'STUDENTST':  2
%               'GED':        3
% OUTPUTS:
%    logL        - The calculated Quasi-Likelihood
%    Rt          - a k x k x t 3 dimesnaional array of conditional correlations
%    likelihoods - a t by 1 vector of quasi likelihoods

%
%
% COMMENTS:
%
%
% Author: Martin Grziska
% based on the DCC code by Kevin Sheppard

[t,k]=size(data);
index=1;
H=zeros(size(data));
stdEstimate =  std(data,1);
m=max(max(archP),max(garchQ));

% for i=1:k
%     if errortype(i) ~= 1
%         gamma(i)=1;
%     else gamma(i)=0;
%     end
% end

gamma=zeros(k,1);

for i=1:k
    univariateparameters=parameters(index:index+archP(i)+o(i)+garchQ(i)+gamma(i));
    if garchtype(i) == 0
        H(:,i) = egarchcore(data(:,i), univariateparameters(1:end-gamma(i)), stdEstimate(:,i), archP(i), o(i), garchQ(i), m , t);
    else
        %         set leverage term to zero if model contains not leverage but
        %         power-parameter
        currentAsymmetricOrder = o(i);
        if garchtype(i) == 4
            currentAsymmetricOrder = 0;
        elseif garchtype(i) == 6
            %     special case: APGARCH contains leverage and power-parameter
            currentAsymmetricOrder = 1;
        end
        [~, H(:,i)] = ADCC_univariate_simulate(univariateparameters, ...
            data(:,i), archP(i), currentAsymmetricOrder, garchQ(i), ...
            garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+gamma(i)+1;
end

stdresid=data./sqrt(H);

a = parameters(index:index+dccP-1);
aNegative = parameters(index+dccP:index+dccP+dccG-1);
b = parameters(index+dccP+dccG:index+dccP+dccG+dccQ-1);
identityMatrix = eye(k);
archMatrices = identityMatrix .* reshape(a, 1, 1, dccP);
asymmetricMatrices = identityMatrix .* reshape(aNegative, 1, 1, dccG);
garchMatrices = identityMatrix .* reshape(b, 1, 1, dccQ);
qBar = cov(stdresid);
negativeQBar = cov(min(stdresid, 0));

[logL, isValid, Rt, likelihoods, Qt] = ...
    dccFilterForRequestedOutputs(nargout, stdresid, H, qBar, ...
    archMatrices, garchMatrices, asymmetricMatrices, negativeQBar);
if ~isValid
    logL = 1e16;
end

