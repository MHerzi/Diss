function [logL, Rt, likelihoods, Qt] = ...
    AGDCC_likelihood_grm(params, stdresid, P, Q, ~)
%AGDCC_LIKELIHOOD_GRM Negative Gaussian asymmetric generalized DCC QML.

    penalty = 1e16;
    [~, seriesCount] = size(stdresid);
    expectedParameterCount = (2 * P + Q) * seriesCount;
    if ~isValidParameterVector(params, expectedParameterCount)
        [logL, Rt, likelihoods, Qt] = invalidOutputs(penalty);
        return
    end

    archParameterCount = P * seriesCount;
    archValues = reshape(params(1:archParameterCount), seriesCount, P);
    asymmetricValues = reshape(params(archParameterCount + ...
        (1:archParameterCount)), seriesCount, P);
    garchValues = reshape(params(2 * archParameterCount + 1:end), ...
        seriesCount, Q);

    archMatrices = diagonalMatrixSequence(archValues);
    asymmetricMatrices = diagonalMatrixSequence(asymmetricValues);
    garchMatrices = diagonalMatrixSequence(garchValues);
    qBar = cov(stdresid);
    negativeQBar = cov(min(stdresid, 0));

    [logL, isValid, Rt, likelihoods, Qt] = ...
        dccFilterForRequestedOutputs(nargout, stdresid, [], qBar, ...
        archMatrices, garchMatrices, asymmetricMatrices, negativeQBar);
    if ~isValid
        logL = penalty;
    end
end

function matrices = diagonalMatrixSequence(values)
    [seriesCount, order] = size(values);
    matrices = zeros(seriesCount, seriesCount, order, 'like', values);
    for lag = 1:order
        matrices(:, :, lag) = diag(values(:, lag));
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
