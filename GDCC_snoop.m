function [Rt_pred,Qt_pred] = GDCC_snoop(params, data, archP, garchQ, ...
    o, garchtype, errortype, dccP, dccQ, forecastCount)

% Data-snooping "forecast": split whole sample in estimation and forecast
% period; use AGDCC parameters from estimation period to predict
% values in forecastin period
%
% USAGE:   [Rt_pred,Qt_pred] = AGDCC_snoop(params, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ)
%
% INPUTS:
%    params      - A 2*P+Q by 1 vector of params of the form [dccPparams; dccQparams, adccPparams]
%    data        - A matrix, t x k of return series  (COMPLETE SET !!!!); "predicted" residuals
%                  from ar_univariate_snoop.m
%    archP       - vector with # of ARCH lags
%    garchQ      - vector with # of GARCH lags
%     o          - vector with # of asymmtric order
%    dccP        - The innovation order of the AGDCC Garch process
%    dccQ        - The GARCH order of the AGDCC estimator
%    errortype   -
%    garchtype   -
%
% OUTPUTS:
%    logL        - The calculated Quasi-Likelihood
%    Rt          - a k x k x t 3 dimesnaional array of conditional correlations
%    likelihoods - a t by 1 vector of quasi likelihoods
%    Qt          - a k x k x t 3 dimesnaional array of conditional
%                  covariances
%
% Author: Martin Grziska, based on a code from Kevin Sheppard 02/26/2010

[t,k]=size(data);
if nargin < 10 || isempty(forecastCount)
    forecastCount = 1;
end
index=1;
H=zeros(size(data));
stdEstimate =  std(data,1);
m=max(max(archP),max(garchQ));

gamma = double(errortype(:) ~= 1);
gamma(errortype(:) == 4) = 2;

for i=1:k
    univariateparams=params(index:index+archP(i)+o(i)+garchQ(i)+gamma(i));
    if garchtype(i)==0
        H(:,i) = egarchcore(data(:,i), univariateparams(1:end-gamma(i)), stdEstimate(:,i), archP(i), o(i), garchQ(i), m , t);
    else
        [~, H(:,i)] = ADCC_univariate_simulate(univariateparams, ...
            data(:,i), archP(i), o(i), garchQ(i), ...
            garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+gamma(i)+1;
end

stdresid=data./sqrt(H);

archParameterCount = dccP * k;
archValues = reshape(params(index:index+archParameterCount-1), k, dccP);
garchStart = index + archParameterCount;
garchValues = reshape(params(garchStart:garchStart+dccQ*k-1), k, dccQ);
archMatrices = dccDiagonalMatrixSequence(archValues);
garchMatrices = dccDiagonalMatrixSequence(garchValues);
emptyMatrices = zeros(k, k, 0);
[Rt_pred, Qt_pred] = dccCorrelationForecastPath(stdresid, ...
    cov(stdresid), archMatrices, garchMatrices, emptyMatrices, ...
    emptyMatrices, forecastCount);
