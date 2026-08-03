function [RtForecast, QtForecast] = forecastPath( ...
    stdResiduals, qBar, archMatrices, garchMatrices, ...
    asymmetricMatrices, negativeQBar, forecastCount)
%FORECASTPATH Return recent one-step DCC forecasts.
%   The function filters the supplied residual history once and returns the
%   final FORECASTCOUNT states, including the one-step-ahead state after
%   the final residual. Only the Q lags required by the recursion are kept.

    validateattributes(stdResiduals, {'double'}, ...
        {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'stdResiduals');
    validateattributes(forecastCount, {'double'}, ...
        {'scalar', 'integer', 'positive'}, mfilename, 'forecastCount');
    [observationCount, seriesCount] = size(stdResiduals);
    if forecastCount > observationCount + 1
        error('Diss:DCC:ForecastCountExceedsHistory', ...
            'forecastCount cannot exceed the number of filtered states.');
    end

    archOrder = size(archMatrices, 3);
    garchOrder = size(garchMatrices, 3);
    asymmetricOrder = size(asymmetricMatrices, 3);
    [archDiagonals, archIsDiagonal] = ...
        extractDiagonalSequence(archMatrices);
    [garchDiagonals, garchIsDiagonal] = ...
        extractDiagonalSequence(garchMatrices);
    [asymmetricDiagonals, asymmetricIsDiagonal] = ...
        extractDiagonalSequence(asymmetricMatrices);
    useDiagonalRecursion = archIsDiagonal && garchIsDiagonal && ...
        asymmetricIsDiagonal;

    qIntercept = qBar;
    for lag = 1:archOrder
        qIntercept = subtractPersistence(qIntercept, qBar, ...
            archMatrices(:, :, lag), archDiagonals(:, lag), ...
            useDiagonalRecursion);
    end
    for lag = 1:garchOrder
        qIntercept = subtractPersistence(qIntercept, qBar, ...
            garchMatrices(:, :, lag), garchDiagonals(:, lag), ...
            useDiagonalRecursion);
    end
    for lag = 1:asymmetricOrder
        qIntercept = subtractPersistence(qIntercept, negativeQBar, ...
            asymmetricMatrices(:, :, lag), ...
            asymmetricDiagonals(:, lag), useDiagonalRecursion);
    end
    qIntercept = (qIntercept + qIntercept') / 2;

    stateCount = observationCount + 1;
    firstReturnedState = stateCount - forecastCount + 1;
    RtForecast = zeros(seriesCount, seriesCount, forecastCount, 'like', qBar);
    QtForecast = zeros(seriesCount, seriesCount, forecastCount, 'like', qBar);
    qBufferLength = max(garchOrder, 1);
    qBuffer = repmat(qBar, 1, 1, qBufferLength);
    if asymmetricOrder > 0
        negativeResiduals = min(stdResiduals, 0);
    else
        negativeResiduals = [];
    end

    for state = 1:stateCount
        currentQ = qIntercept;
        for lag = 1:archOrder
            previousObservation = state - lag;
            if previousObservation > 0
                residual = stdResiduals(previousObservation, :);
                currentQ = addInnovation(currentQ, residual, ...
                    archMatrices(:, :, lag), archDiagonals(:, lag), ...
                    useDiagonalRecursion);
            end
        end
        for lag = 1:asymmetricOrder
            previousObservation = state - lag;
            if previousObservation > 0
                residual = negativeResiduals(previousObservation, :);
                currentQ = addInnovation(currentQ, residual, ...
                    asymmetricMatrices(:, :, lag), ...
                    asymmetricDiagonals(:, lag), useDiagonalRecursion);
            end
        end
        for lag = 1:garchOrder
            previousState = state - lag;
            if previousState > 0
                bufferIndex = mod(previousState - 1, qBufferLength) + 1;
                previousQ = qBuffer(:, :, bufferIndex);
            else
                previousQ = qBar;
            end
            if useDiagonalRecursion
                diagonal = garchDiagonals(:, lag);
                currentQ = currentQ + previousQ .* (diagonal * diagonal');
            else
                matrix = garchMatrices(:, :, lag);
                currentQ = currentQ + matrix' * previousQ * matrix;
            end
        end

        currentQ = (currentQ + currentQ') / 2;
        qDiagonal = diag(currentQ);
        if any(~isfinite(qDiagonal) | qDiagonal <= 0)
            error('Diss:DCC:InvalidForecastCovariance', ...
                'The DCC recursion produced an invalid covariance state.');
        end
        scale = sqrt(qDiagonal);
        currentR = currentQ ./ (scale * scale');
        currentR = (currentR + currentR') / 2;
        currentR(1:seriesCount + 1:end) = 1;
        [~, cholStatus] = chol(currentR, 'lower');
        if cholStatus ~= 0
            error('Diss:DCC:NonPositiveDefiniteForecast', ...
                'The DCC recursion produced a non-positive-definite state.');
        end

        if state >= firstReturnedState
            outputIndex = state - firstReturnedState + 1;
            RtForecast(:, :, outputIndex) = currentR;
            QtForecast(:, :, outputIndex) = currentQ;
        end
        bufferIndex = mod(state - 1, qBufferLength) + 1;
        qBuffer(:, :, bufferIndex) = currentQ;
    end
end

function currentQ = subtractPersistence( ...
    currentQ, unconditionalQ, matrix, diagonal, useDiagonalRecursion)
    if useDiagonalRecursion
        currentQ = currentQ - unconditionalQ .* (diagonal * diagonal');
    else
        currentQ = currentQ - matrix' * unconditionalQ * matrix;
    end
end

function currentQ = addInnovation( ...
    currentQ, residual, matrix, diagonal, useDiagonalRecursion)
    if useDiagonalRecursion
        transformedResidual = residual .* diagonal';
    else
        transformedResidual = residual * matrix;
    end
    currentQ = currentQ + transformedResidual' * transformedResidual;
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
