function tests = TestCorrelatedGaussianSimulation
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
end

function testCorrelationPrecedesMarginalTransformation(testCase)
    means = [0.02, -0.01, 0.04];
    variances = [0.04, 0.09, 0.16];
    correlation = [1, 0.45, -0.20; 0.45, 1, 0.30; -0.20, 0.30, 1];
    simulationCount = 1000;

    previousState = rng(5104, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    lowerFactor = chol(correlation, 'lower');
    expected = means + randn(simulationCount, 3) * lowerFactor' .* ...
        sqrt(variances);

    rng(5104, 'twister');
    actual = simulateCorrelatedGaussianReturns( ...
        means, variances, correlation, simulationCount);
    verifyEqual(testCase, actual, expected, 'AbsTol', 1e-14);
end

function testSimulatedMomentsMatchTarget(testCase)
    means = [0.02, -0.01, 0.04];
    variances = [0.04, 0.09, 0.16];
    correlation = [1, 0.45, -0.20; 0.45, 1, 0.30; -0.20, 0.30, 1];
    targetCovariance = correlation .* ...
        (sqrt(variances)' * sqrt(variances));

    previousState = rng(9921, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    actual = simulateCorrelatedGaussianReturns( ...
        means, variances, correlation, 200000);

    verifyEqual(testCase, mean(actual, 1), means, 'AbsTol', 3e-3);
    verifyEqual(testCase, cov(actual), targetCovariance, 'AbsTol', 3e-3);
end

function testRejectsNonPositiveDefiniteCorrelation(testCase)
    invalidCorrelation = [1, 1.2; 1.2, 1];
    verifyError(testCase, @() simulateCorrelatedGaussianReturns( ...
        [0, 0], [1, 1], invalidCorrelation, 100), ...
        'Diss:VaR:NonPositiveDefiniteCorrelation');
end
