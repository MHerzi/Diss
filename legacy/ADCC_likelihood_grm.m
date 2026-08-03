function [logL, Rt, likelihoods, Qt] = ...
    ADCC_likelihood_grm(params, stdresid, P, Q, ~)
%ADCC_LIKELIHOOD_GRM Negative Gaussian asymmetric DCC quasi-likelihood.

    penalty = 1e16;
    [~, seriesCount] = size(stdresid);
    if ~isValidParameterVector(params, 2 * P + Q)
        [logL, Rt, likelihoods, Qt] = invalidOutputs(penalty);
        return
    end

    identityMatrix = eye(seriesCount);
    archMatrices = identityMatrix .* reshape(params(1:P), 1, 1, P);
    asymmetricMatrices = identityMatrix .* ...
        reshape(params(P + (1:P)), 1, 1, P);
    garchMatrices = identityMatrix .* ...
        reshape(params(2 * P + (1:Q)), 1, 1, Q);
    qBar = cov(stdresid);
    negativeQBar = cov(min(stdresid, 0));

    [logL, isValid, Rt, likelihoods, Qt] = ...
        dccFilterForRequestedOutputs(nargout, stdresid, [], qBar, ...
        archMatrices, garchMatrices, asymmetricMatrices, negativeQBar);
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
