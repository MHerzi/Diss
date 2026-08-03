function tests = TestDccCorrelationFilter
% Regression tests for the modernized DCC numerical kernel.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
    testCase.TestData.RepositoryRoot = repositoryRoot;
end

function testScalarDccMatchesLegacyImplementation(testCase)
    [stdResiduals, qBar] = syntheticResiduals();
    parameters = [0.12; 0.78];
    identityMatrix = eye(size(stdResiduals, 2));
    archMatrices = identityMatrix * parameters(1);
    garchMatrices = identityMatrix * parameters(2);
    emptyMatrices = zeros(size(identityMatrix, 1), ...
        size(identityMatrix, 2), 0);

    [expectedLogL, expectedRt, expectedLikelihoods, expectedQt] = ...
        legacyFilter(stdResiduals, qBar, archMatrices, garchMatrices, ...
        emptyMatrices, emptyMatrices);
    [actualLogL, actualRt, actualLikelihoods, actualQt] = ...
        dcc_mvgarch_likelihood_grm(parameters, stdResiduals, 1, 1, 1e-6);

    verifyNumericalEquivalence(testCase, actualLogL, actualRt, ...
        actualLikelihoods, actualQt, expectedLogL, expectedRt, ...
        expectedLikelihoods, expectedQt);
end

function testAsymmetricDccMatchesLegacyImplementation(testCase)
    [stdResiduals, qBar] = syntheticResiduals();
    parameters = [0.10; 0.04; 0.80];
    identityMatrix = eye(size(stdResiduals, 2));
    archMatrices = identityMatrix * parameters(1);
    asymmetricMatrices = identityMatrix * parameters(2);
    garchMatrices = identityMatrix * parameters(3);
    negativeQBar = cov(min(stdResiduals, 0));

    [expectedLogL, expectedRt, expectedLikelihoods, expectedQt] = ...
        legacyFilter(stdResiduals, qBar, archMatrices, garchMatrices, ...
        asymmetricMatrices, negativeQBar);
    [actualLogL, actualRt, actualLikelihoods, actualQt] = ...
        ADCC_likelihood_grm(parameters, stdResiduals, 1, 1, 1e-6);

    verifyNumericalEquivalence(testCase, actualLogL, actualRt, ...
        actualLikelihoods, actualQt, expectedLogL, expectedRt, ...
        expectedLikelihoods, expectedQt);
end

function testGeneralizedDccMatchesLegacyImplementation(testCase)
    [stdResiduals, qBar] = syntheticResiduals();
    parameters = [0.08; 0.10; 0.12; 0.76; 0.78; 0.80];
    archMatrices = diag(parameters(1:3));
    garchMatrices = diag(parameters(4:6));
    emptyMatrices = zeros(3, 3, 0);

    [expectedLogL, expectedRt, expectedLikelihoods, expectedQt] = ...
        legacyFilter(stdResiduals, qBar, archMatrices, garchMatrices, ...
        emptyMatrices, emptyMatrices);
    [actualLogL, actualRt, actualLikelihoods, actualQt] = ...
        GDCC_likelihood(parameters, stdResiduals, 1, 1, 1e-6);

    verifyNumericalEquivalence(testCase, actualLogL, actualRt, ...
        actualLikelihoods, actualQt, expectedLogL, expectedRt, ...
        expectedLikelihoods, expectedQt);
end

function testAsymmetricGeneralizedDccMatchesLegacyImplementation(testCase)
    [stdResiduals, qBar] = syntheticResiduals();
    parameters = [0.08; 0.10; 0.12; 0.02; 0.03; 0.04; ...
        0.74; 0.76; 0.78];
    archMatrices = diag(parameters(1:3));
    asymmetricMatrices = diag(parameters(4:6));
    garchMatrices = diag(parameters(7:9));
    negativeQBar = cov(min(stdResiduals, 0));

    [expectedLogL, expectedRt, expectedLikelihoods, expectedQt] = ...
        legacyFilter(stdResiduals, qBar, archMatrices, garchMatrices, ...
        asymmetricMatrices, negativeQBar);
    [actualLogL, actualRt, actualLikelihoods, actualQt] = ...
        AGDCC_likelihood_grm(parameters, stdResiduals, 1, 1, 1e-6);

    verifyNumericalEquivalence(testCase, actualLogL, actualRt, ...
        actualLikelihoods, actualQt, expectedLogL, expectedRt, ...
        expectedLikelihoods, expectedQt);
end

function testInvalidParametersReturnFinitePenalty(testCase)
    [stdResiduals, ~] = syntheticResiduals();
    actualLogL = dcc_mvgarch_likelihood_grm([2; 2], ...
        stdResiduals, 1, 1, 1e-6);
    verifyEqual(testCase, actualLogL, 1e9);
end

function testMarginalVarianceTermsMatchLegacyFormula(testCase)
    [stdResiduals, qBar] = syntheticResiduals();
    conditionalVariances = 0.05 + abs(stdResiduals) / 10;
    identityMatrix = eye(size(stdResiduals, 2));
    archMatrices = identityMatrix * 0.12;
    garchMatrices = identityMatrix * 0.78;
    emptyMatrices = zeros(size(identityMatrix, 1), ...
        size(identityMatrix, 2), 0);

    [correlationLogL, expectedRt, correlationLikelihoods, expectedQt] = ...
        legacyFilter(stdResiduals, qBar, archMatrices, garchMatrices, ...
        emptyMatrices, emptyMatrices);
    marginalTerms = 0.5 * (size(stdResiduals, 2) * log(2*pi) + ...
        sum(log(conditionalVariances), 2));
    expectedLogL = correlationLogL + sum(marginalTerms);
    expectedLikelihoods = correlationLikelihoods + marginalTerms;

    [actualLogL, isValid, actualRt, actualLikelihoods, actualQt] = ...
        dccCorrelationFilter(stdResiduals, conditionalVariances, qBar, ...
        archMatrices, garchMatrices, emptyMatrices, emptyMatrices);

    verifyTrue(testCase, isValid);
    verifyNumericalEquivalence(testCase, actualLogL, actualRt, ...
        actualLikelihoods, actualQt, expectedLogL, expectedRt, ...
        expectedLikelihoods, expectedQt);
end

function [stdResiduals, qBar] = syntheticResiduals()
    previousState = rng(91827, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    stdResiduals = randn(250, 3);
    stdResiduals(:, 2) = 0.35 * stdResiduals(:, 1) + ...
        sqrt(1 - 0.35^2) * stdResiduals(:, 2);
    qBar = cov(stdResiduals);
end

function verifyNumericalEquivalence(testCase, actualLogL, actualRt, ...
    actualLikelihoods, actualQt, expectedLogL, expectedRt, ...
    expectedLikelihoods, expectedQt)
    verifyEqual(testCase, actualLogL, expectedLogL, 'AbsTol', 1e-10);
    verifyEqual(testCase, actualRt, expectedRt, 'AbsTol', 1e-10);
    verifyEqual(testCase, actualLikelihoods(:), expectedLikelihoods(:), ...
        'AbsTol', 1e-10);
    verifyEqual(testCase, actualQt, expectedQt, 'AbsTol', 1e-10);
end

function [logL, Rt, likelihoods, Qt] = legacyFilter(stdResiduals, qBar, ...
    archMatrices, garchMatrices, asymmetricMatrices, negativeQBar)
% Direct transcription of the previous allocation/inv/det implementation.
    [observationCount, seriesCount] = size(stdResiduals);
    archOrder = size(archMatrices, 3);
    garchOrder = size(garchMatrices, 3);
    asymmetricOrder = size(asymmetricMatrices, 3);
    lagCount = max([archOrder, garchOrder, asymmetricOrder, 1]);

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
    QtPadded = repmat(qBar, 1, 1, observationCount + lagCount);
    RtPadded = zeros(seriesCount, seriesCount, observationCount + lagCount);
    likelihoods = zeros(observationCount, 1);

    for observation = 1:observationCount
        paddedIndex = observation + lagCount;
        currentQ = qIntercept;
        for lag = 1:archOrder
            residual = paddedResiduals(paddedIndex - lag, :);
            matrix = archMatrices(:, :, lag);
            currentQ = currentQ + matrix' * (residual' * residual) * matrix;
        end
        for lag = 1:asymmetricOrder
            residual = negativeResiduals(paddedIndex - lag, :);
            matrix = asymmetricMatrices(:, :, lag);
            currentQ = currentQ + matrix' * (residual' * residual) * matrix;
        end
        for lag = 1:garchOrder
            matrix = garchMatrices(:, :, lag);
            currentQ = currentQ + matrix' * ...
                QtPadded(:, :, paddedIndex - lag) * matrix;
        end
        QtPadded(:, :, paddedIndex) = currentQ;
        scale = sqrt(diag(currentQ));
        currentR = currentQ ./ (scale * scale');
        currentR(1:seriesCount + 1:end) = 1;
        RtPadded(:, :, paddedIndex) = currentR;
        residual = paddedResiduals(paddedIndex, :);
        likelihoods(observation) = 0.5 * (log(det(currentR)) + ...
            residual * inv(currentR) * residual'); %#ok<MINV>
    end

    Qt = QtPadded(:, :, lagCount + (1:observationCount));
    Rt = RtPadded(:, :, lagCount + (1:observationCount));
    logL = sum(likelihoods);
end
