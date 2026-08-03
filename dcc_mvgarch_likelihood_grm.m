function [logL, Rt, likelihoods, Qt] = ...
    dcc_mvgarch_likelihood_grm(params, stdresid, P, Q, ~)
%DCC_MVGARCH_LIKELIHOOD_GRM Negative Gaussian DCC quasi-likelihood.
%   Parameter values are square roots of the conventional DCC
%   coefficients, preserving the original model parameterization.

    penalty = 1e9;
    [observationCount, seriesCount] = size(stdresid); %#ok<ASGLU>
    if ~isValidParameterVector(params, P + Q)
        [logL, Rt, likelihoods, Qt] = invalidOutputs(penalty);
        return
    end

    identityMatrix = eye(seriesCount);
    archMatrices = identityMatrix .* reshape(params(1:P), 1, 1, P);
    garchMatrices = identityMatrix .* ...
        reshape(params(P + (1:Q)), 1, 1, Q);
    emptyMatrices = zeros(seriesCount, seriesCount, 0);

    [logL, isValid, Rt, likelihoods, Qt] = ...
        dccFilterForRequestedOutputs(nargout, stdresid, [], ...
        cov(stdresid), archMatrices, garchMatrices, ...
        emptyMatrices, emptyMatrices);
    if ~isValid
        logL = penalty;
    end
end

function tf = isValidParameterVector(params, expectedCount)
    tf = isnumeric(params) && isreal(params) && ...
        numel(params) == expectedCount && all(isfinite(params(:)));
end

function [logL, Rt, likelihoods, Qt] = invalidOutputs(penalty)
    logL = penalty;
    Rt = [];
    likelihoods = [];
    Qt = [];
end
