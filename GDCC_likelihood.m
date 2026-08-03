function [logL, Rt, likelihoods, Qt] = ...
    GDCC_likelihood(params, stdresid, P, Q, ~)
%GDCC_LIKELIHOOD Negative Gaussian generalized DCC quasi-likelihood.

    penalty = 1e16;
    [~, seriesCount] = size(stdresid);
    if ~isValidParameterVector(params, (P + Q) * seriesCount)
        [logL, Rt, likelihoods, Qt] = invalidOutputs(penalty);
        return
    end

    archParameterCount = P * seriesCount;
    archValues = reshape(params(1:archParameterCount), seriesCount, P);
    garchValues = reshape(params(archParameterCount + 1:end), ...
        seriesCount, Q);
    archMatrices = diagonalMatrixSequence(archValues);
    garchMatrices = diagonalMatrixSequence(garchValues);
    emptyMatrices = zeros(seriesCount, seriesCount, 0);

    [logL, isValid, Rt, likelihoods, Qt] = ...
        dccFilterForRequestedOutputs(nargout, stdresid, [], ...
        cov(stdresid), archMatrices, garchMatrices, ...
        emptyMatrices, emptyMatrices);
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
