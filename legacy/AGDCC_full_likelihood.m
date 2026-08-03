function [logL, Rt, likelihoods, Qt] = AGDCC_full_likelihood(params, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ, dccG)

% PURPOSE:
%        Restricted likelihood for use in the AGDCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC params
%
% USAGE:
%        [logL, Rt, likelihoods, Qt] = AGDCC_full_likelihood(params, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ)
%
% INPUTS:
%    params      - A 2*P+Q by 1 vector of params of the form [dccPparams; dccQparams, adccPparams]
%    data        - A matrix, t x k of return series
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
index=1;
H=zeros(size(data));
stdEstimate =  std(data,1);
m=max(max(archP),max(garchQ));

gamma = double(errortype(:) ~= 1);

for i=1:k
    univariateparams=params(index:index+archP(i)+o(i)+garchQ(i)+gamma(i));
    if garchtype(i) == 0
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
asymmetricStart = index + archParameterCount;
asymmetricValues = reshape(params( ...
    asymmetricStart:asymmetricStart+dccG*k-1), k, dccG);
garchStart = asymmetricStart + dccG * k;
garchValues = reshape(params(garchStart:garchStart+dccQ*k-1), k, dccQ);

archMatrices = diagonalMatrixSequence(archValues);
asymmetricMatrices = diagonalMatrixSequence(asymmetricValues);
garchMatrices = diagonalMatrixSequence(garchValues);
qBar = cov(stdresid);
negativeQBar = cov(min(stdresid, 0));

[logL, isValid, Rt, likelihoods, Qt] = ...
    dccFilterForRequestedOutputs(nargout, stdresid, H, qBar, ...
    archMatrices, garchMatrices, asymmetricMatrices, negativeQBar);
if ~isValid
    logL = 1e16;
end
end

function matrices = diagonalMatrixSequence(values)
    [seriesCount, order] = size(values);
    matrices = zeros(seriesCount, seriesCount, order, 'like', values);
    for lag = 1:order
        matrices(:, :, lag) = diag(values(:, lag));
    end
end

