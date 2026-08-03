function [negativeLogLikelihood, isValid, Rt, likelihoods, Qt] = ...
    filter(stdResiduals, conditionalVariances, qBar, ...
    archMatrices, garchMatrices, asymmetricMatrices, negativeQBar)
%FILTER Filter a (generalized, asymmetric) DCC process.
%   This internal numerical kernel uses Cholesky factors instead of
%   explicit matrix inverses and determinants. When RT and QT are not
%   requested, only the Q lags needed by the recursion are retained.

    validateattributes(stdResiduals, {'double'}, ...
        {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'stdResiduals');
    validateattributes(qBar, {'double'}, ...
        {'2d', 'real', 'finite', 'square'}, mfilename, 'qBar');

    [observationCount, seriesCount] = size(stdResiduals);
    if ~isequal(size(qBar), [seriesCount, seriesCount])
        error('Diss:DCC:InvalidQBarSize', ...
            'qBar must have one row and column per residual series.');
    end

    if isempty(conditionalVariances)
        includeMarginalLikelihood = false;
    else
        validateattributes(conditionalVariances, {'double'}, ...
            {'2d', 'real', 'finite', 'positive', ...
             'size', size(stdResiduals)}, mfilename, ...
            'conditionalVariances');
        includeMarginalLikelihood = true;
    end

    validateMatrixSequence(archMatrices, seriesCount, 'archMatrices');
    validateMatrixSequence(garchMatrices, seriesCount, 'garchMatrices');
    validateMatrixSequence(asymmetricMatrices, seriesCount, ...
        'asymmetricMatrices');

    asymmetricOrder = size(asymmetricMatrices, 3);
    if asymmetricOrder == 0
        negativeQBar = zeros(seriesCount, 'like', qBar);
    else
        validateattributes(negativeQBar, {'double'}, ...
            {'2d', 'real', 'finite', 'square', ...
             'size', size(qBar)}, mfilename, 'negativeQBar');
    end

    archOrder = size(archMatrices, 3);
    garchOrder = size(garchMatrices, 3);
    if archOrder == 0 && garchOrder == 0 && asymmetricOrder == 0
        error('Diss:DCC:MissingDynamics', ...
            'At least one DCC parameter matrix is required.');
    end

    [archDiagonals, archIsDiagonal] = ...
        extractDiagonalSequence(archMatrices);
    [garchDiagonals, garchIsDiagonal] = ...
        extractDiagonalSequence(garchMatrices);
    [asymmetricDiagonals, asymmetricIsDiagonal] = ...
        extractDiagonalSequence(asymmetricMatrices);
    useDiagonalRecursion = archIsDiagonal && garchIsDiagonal && ...
        asymmetricIsDiagonal;

    returnRt = nargout >= 3;
    returnLikelihoods = nargout >= 4;
    returnQt = nargout >= 5;

    if returnRt
        Rt = zeros(seriesCount, seriesCount, observationCount, 'like', qBar);
    else
        Rt = [];
    end
    if returnLikelihoods
        likelihoods = zeros(observationCount, 1, 'like', stdResiduals);
    else
        likelihoods = [];
    end
    if returnQt
        Qt = zeros(seriesCount, seriesCount, observationCount, 'like', qBar);
    else
        Qt = [];
    end

    qIntercept = qBar;
    for lag = 1:archOrder
        if useDiagonalRecursion
            diagonal = archDiagonals(:, lag);
            qIntercept = qIntercept - qBar .* (diagonal * diagonal');
        else
            matrix = archMatrices(:, :, lag);
            qIntercept = qIntercept - matrix' * qBar * matrix;
        end
    end
    for lag = 1:garchOrder
        if useDiagonalRecursion
            diagonal = garchDiagonals(:, lag);
            qIntercept = qIntercept - qBar .* (diagonal * diagonal');
        else
            matrix = garchMatrices(:, :, lag);
            qIntercept = qIntercept - matrix' * qBar * matrix;
        end
    end
    for lag = 1:asymmetricOrder
        if useDiagonalRecursion
            diagonal = asymmetricDiagonals(:, lag);
            qIntercept = qIntercept - ...
                negativeQBar .* (diagonal * diagonal');
        else
            matrix = asymmetricMatrices(:, :, lag);
            qIntercept = qIntercept - matrix' * negativeQBar * matrix;
        end
    end
    qIntercept = (qIntercept + qIntercept') / 2;

    qBufferLength = max(garchOrder, 1);
    qBuffer = repmat(qBar, 1, 1, qBufferLength);
    if asymmetricOrder > 0
        negativeResiduals = min(stdResiduals, 0);
    else
        negativeResiduals = [];
    end
    negativeLogLikelihood = 0;
    isValid = true;
    gaussianConstant = seriesCount * log(2 * pi);

    for observation = 1:observationCount
        currentQ = qIntercept;

        for lag = 1:archOrder
            previousObservation = observation - lag;
            if previousObservation > 0
                if useDiagonalRecursion
                    transformedResidual = ...
                        stdResiduals(previousObservation, :) .* ...
                        archDiagonals(:, lag)';
                else
                    transformedResidual = ...
                        stdResiduals(previousObservation, :) * ...
                        archMatrices(:, :, lag);
                end
                currentQ = currentQ + ...
                    transformedResidual' * transformedResidual;
            end
        end

        for lag = 1:asymmetricOrder
            previousObservation = observation - lag;
            if previousObservation > 0
                if useDiagonalRecursion
                    transformedResidual = ...
                        negativeResiduals(previousObservation, :) .* ...
                        asymmetricDiagonals(:, lag)';
                else
                    transformedResidual = ...
                        negativeResiduals(previousObservation, :) * ...
                        asymmetricMatrices(:, :, lag);
                end
                currentQ = currentQ + ...
                    transformedResidual' * transformedResidual;
            end
        end

        for lag = 1:garchOrder
            previousObservation = observation - lag;
            if previousObservation > 0
                bufferIndex = mod(previousObservation - 1, ...
                    qBufferLength) + 1;
                previousQ = qBuffer(:, :, bufferIndex);
            else
                previousQ = qBar;
            end
            if useDiagonalRecursion
                diagonal = garchDiagonals(:, lag);
                currentQ = currentQ + previousQ .* ...
                    (diagonal * diagonal');
            else
                matrix = garchMatrices(:, :, lag);
                currentQ = currentQ + matrix' * previousQ * matrix;
            end
        end

        currentQ = (currentQ + currentQ') / 2;
        qDiagonal = diag(currentQ);
        if any(~isfinite(qDiagonal)) || any(qDiagonal <= 0)
            isValid = false;
            break
        end

        qScale = sqrt(qDiagonal);
        currentR = currentQ ./ (qScale * qScale');
        currentR = (currentR + currentR') / 2;
        currentR(1:seriesCount + 1:end) = 1;

        [lowerFactor, cholStatus] = chol(currentR, 'lower');
        if cholStatus ~= 0 || any(abs(currentR(:)) > 1 + 1e-10)
            isValid = false;
            break
        end

        standardizedObservation = stdResiduals(observation, :)';
        solvedObservation = lowerFactor \ standardizedObservation;
        contribution = 2 * sum(log(diag(lowerFactor))) + ...
            dot(solvedObservation, solvedObservation);
        if includeMarginalLikelihood
            contribution = contribution + gaussianConstant + ...
                sum(log(conditionalVariances(observation, :)));
        end

        if ~isfinite(contribution) || ~isreal(contribution)
            isValid = false;
            break
        end

        negativeLogLikelihood = negativeLogLikelihood + contribution;
        if returnLikelihoods
            likelihoods(observation) = contribution / 2;
        end
        if returnRt
            Rt(:, :, observation) = currentR;
        end
        if returnQt
            Qt(:, :, observation) = currentQ;
        end

        bufferIndex = mod(observation - 1, qBufferLength) + 1;
        qBuffer(:, :, bufferIndex) = currentQ;
    end

    negativeLogLikelihood = negativeLogLikelihood / 2;
    if ~isValid
        negativeLogLikelihood = NaN;
        if returnLikelihoods
            likelihoods(:) = NaN;
        end
        if returnRt
            Rt(:, :, observation:end) = NaN;
        end
        if returnQt
            Qt(:, :, observation:end) = NaN;
        end
    end
end

function [diagonals, isDiagonal] = extractDiagonalSequence(matrices)
    seriesCount = size(matrices, 1);
    order = size(matrices, 3);
    diagonals = zeros(seriesCount, order, 'like', matrices);
    isDiagonal = true;
    for lag = 1:order
        matrix = matrices(:, :, lag);
        diagonals(:, lag) = diag(matrix);
        matrix(1:seriesCount + 1:end) = 0;
        if any(matrix(:) ~= 0)
            isDiagonal = false;
            return
        end
    end
end

function validateMatrixSequence(matrices, seriesCount, argumentName)
    validateattributes(matrices, {'double'}, ...
        {'real', 'finite'}, mfilename, argumentName);
    matrixSize = size(matrices);
    if numel(matrixSize) < 3
        matrixSize(3) = 1;
    end
    if matrixSize(1) ~= seriesCount || matrixSize(2) ~= seriesCount
        error('Diss:DCC:InvalidParameterMatrixSize', ...
            '%s must contain square matrices matching the series count.', ...
            argumentName);
    end
end
