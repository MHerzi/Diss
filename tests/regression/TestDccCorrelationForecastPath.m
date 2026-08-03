function tests = TestDccCorrelationForecastPath
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    repositoryRoot = fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))));
    addpath(repositoryRoot);
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
        fullfile(repositoryRoot, 'legacy')));
    addpath(fullfile(repositoryRoot, 'src'));
end

function testRecentForecastsMatchLegacyFullAllocation(testCase)
    previousState = rng(8853, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    stdResiduals = randn(300, 4);
    qBar = cov(stdResiduals);
    identityMatrix = eye(4);
    archMatrices = cat(3, 0.08 * identityMatrix, 0.04 * identityMatrix);
    garchMatrices = 0.78 * identityMatrix;
    asymmetricMatrices = cat(3, ...
        0.02 * identityMatrix, 0.01 * identityMatrix);
    negativeQBar = cov(min(stdResiduals, 0));
    forecastCount = 12;

    [expectedRt, expectedQt] = legacyForecastPath(stdResiduals, qBar, ...
        archMatrices, garchMatrices, asymmetricMatrices, ...
        negativeQBar, forecastCount);
    [actualRt, actualQt] = dccCorrelationForecastPath(stdResiduals, qBar, ...
        archMatrices, garchMatrices, asymmetricMatrices, ...
        negativeQBar, forecastCount);

    verifyEqual(testCase, actualRt, expectedRt, 'AbsTol', 1e-11);
    verifyEqual(testCase, actualQt, expectedQt, 'AbsTol', 1e-11);
end

function testGeneralizedDiagonalForecastIsPositiveDefinite(testCase)
    previousState = rng(1154, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    stdResiduals = randn(200, 3);
    qBar = cov(stdResiduals);
    archMatrices = dccDiagonalMatrixSequence([.08; .10; .12]);
    garchMatrices = dccDiagonalMatrixSequence([.76; .78; .80]);
    emptyMatrices = zeros(3, 3, 0);
    [Rt, ~] = dccCorrelationForecastPath(stdResiduals, qBar, ...
        archMatrices, garchMatrices, emptyMatrices, emptyMatrices, 5);
    for page = 1:size(Rt, 3)
        [~, status] = chol(Rt(:, :, page));
        verifyEqual(testCase, status, 0);
    end
end

function testNamespacedForecastMatchesCompatibilityFunction(testCase)
    previousState = rng(7004, 'twister');
    cleanup = onCleanup(@() rng(previousState));
    residuals = randn(180, 3);
    qBar = cov(residuals);
    archMatrices = 0.1 * eye(3);
    garchMatrices = 0.8 * eye(3);
    emptyMatrices = zeros(3, 3, 0);

    [expectedR, expectedQ] = dccCorrelationForecastPath( ...
        residuals, qBar, archMatrices, garchMatrices, ...
        emptyMatrices, emptyMatrices, 7);
    [actualR, actualQ] = diss.dependence.correlation.forecastPath( ...
        residuals, qBar, archMatrices, garchMatrices, ...
        emptyMatrices, emptyMatrices, 7);

    verifyEqual(testCase, actualR, expectedR, 'AbsTol', 0);
    verifyEqual(testCase, actualQ, expectedQ, 'AbsTol', 0);
end

function [RtForecast, QtForecast] = legacyForecastPath( ...
    stdResiduals, qBar, archMatrices, garchMatrices, ...
    asymmetricMatrices, negativeQBar, forecastCount)
    [observationCount, seriesCount] = size(stdResiduals);
    archOrder = size(archMatrices, 3);
    garchOrder = size(garchMatrices, 3);
    asymmetricOrder = size(asymmetricMatrices, 3);
    lagCount = max([archOrder, garchOrder, asymmetricOrder, 1]);
    stateCount = observationCount + 1;

    qIntercept = qBar;
    for lag = 1:archOrder
        matrix = archMatrices(:, :, lag);
        qIntercept = qIntercept - matrix' * qBar * matrix;
    end
    for lag = 1:garchOrder
        matrix = garchMatrices(:, :, lag);
        qIntercept = qIntercept - matrix' * qBar * matrix;
    end
    for lag = 1:asymmetricOrder
        matrix = asymmetricMatrices(:, :, lag);
        qIntercept = qIntercept - matrix' * negativeQBar * matrix;
    end

    paddedResiduals = [zeros(lagCount, seriesCount); stdResiduals];
    negativeResiduals = min(paddedResiduals, 0);
    Qt = repmat(qBar, 1, 1, stateCount + lagCount);
    Rt = zeros(seriesCount, seriesCount, stateCount + lagCount);
    for state = 1:stateCount
        paddedState = state + lagCount;
        currentQ = qIntercept;
        for lag = 1:archOrder
            residual = paddedResiduals(paddedState - lag, :);
            matrix = archMatrices(:, :, lag);
            currentQ = currentQ + matrix' * (residual' * residual) * matrix;
        end
        for lag = 1:asymmetricOrder
            residual = negativeResiduals(paddedState - lag, :);
            matrix = asymmetricMatrices(:, :, lag);
            currentQ = currentQ + matrix' * (residual' * residual) * matrix;
        end
        for lag = 1:garchOrder
            matrix = garchMatrices(:, :, lag);
            currentQ = currentQ + matrix' * ...
                Qt(:, :, paddedState - lag) * matrix;
        end
        Qt(:, :, paddedState) = currentQ;
        scale = sqrt(diag(currentQ));
        currentR = currentQ ./ (scale * scale');
        currentR(1:seriesCount + 1:end) = 1;
        Rt(:, :, paddedState) = currentR;
    end
    returnedStates = stateCount - forecastCount + 1:stateCount;
    QtForecast = Qt(:, :, lagCount + returnedStates);
    RtForecast = Rt(:, :, lagCount + returnedStates);
end
