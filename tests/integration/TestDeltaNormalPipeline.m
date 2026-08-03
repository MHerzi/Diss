function tests = TestDeltaNormalPipeline
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))));
    addpath(repositoryRoot);
    addpath(fullfile(repositoryRoot, 'src'));
end

function testFullSampleFitMatchesClosedFormResult(testCase)
    data = [0.01, -0.02; 0.03, 0.01; -0.01, 0.02; 0.02, -0.01];
    config = deltaNormalConfig();
    config.risk.probability = 0.99;

    results = diss.run(data, config);

    weights = [0.5, 0.5];
    expectedMean = mean(data, 1);
    expectedCovariance = cov(data, 0);
    expectedPortfolioMean = expectedMean * weights';
    expectedPortfolioStd = sqrt(weights * expectedCovariance * weights');
    expectedQuantile = expectedPortfolioMean - ...
        sqrt(2) * erfcinv(0.02) * expectedPortfolioStd;

    verifyEqual(testCase, results.Model.AssetMean, expectedMean, ...
        'AbsTol', 1e-15);
    verifyEqual(testCase, results.Model.AssetCovariance, ...
        expectedCovariance, 'AbsTol', 1e-15);
    verifyEqual(testCase, results.Forecast.ReturnQuantile, ...
        expectedQuantile, 'AbsTol', 1e-15);
    verifyEqual(testCase, results.Forecast.LossVaR, ...
        -expectedQuantile, 'AbsTol', 1e-15);
end

function testBacktestUsesOnlyCurrentEstimationWindow(testCase)
    data = [(1:12)', (21:32)'] / 100;
    config = deltaNormalConfig();
    config.execution.mode = "backtest";
    config.backtest.initialWindow = 6;
    config.backtest.forecastHorizon = 2;
    config.backtest.stepSize = 2;

    results = diss.run(data, config);

    expectedMeans = zeros(height(results.Backtest), 1);
    for row = 1:height(results.Backtest)
        trainingData = data( ...
            results.Backtest.EstimationStart(row): ...
            results.Backtest.EstimationEnd(row), :);
        expectedMeans(row) = mean(trainingData * [0.5; 0.5]);
    end

    verifyEqual(testCase, results.Backtest.ForecastMean, ...
        expectedMeans, 'AbsTol', 1e-15);
    verifyEqual(testCase, results.Backtest.RealizedReturn, ...
        data(results.Backtest.ObservationIndex, :) * [0.5; 0.5], ...
        'AbsTol', 1e-15);
end

function testConfiguredPortfolioWeightsAreUsedConsistently(testCase)
    data = [0.01, 0.04; 0.03, -0.02; -0.01, 0.01; 0.02, 0.03];
    config = deltaNormalConfig();
    config.risk.portfolioWeights = [0.75, 0.25];

    results = diss.run(data, config);

    verifyEqual(testCase, results.Model.PortfolioWeights, [0.75, 0.25]);
    verifyEqual(testCase, results.Model.PortfolioMean, ...
        mean(data, 1) * [0.75; 0.25], 'AbsTol', 1e-15);
end

function testInvalidPortfolioWeightsFailBeforeEstimation(testCase)
    config = deltaNormalConfig();
    config.risk.portfolioWeights = [0.8, 0.3];

    verifyError(testCase, @() diss.run(randn(20, 2), config), ...
        'diss:config:InvalidPortfolioWeights');
end

function testUnsupportedModelFailsExplicitly(testCase)
    config = diss.config.defaults();
    config.model.kind = "vineCopula";
    config.dependence.kind = "vine";

    verifyError(testCase, @() diss.run(randn(20, 2), config), ...
        'diss:config:UnsupportedModelAdapter');
end

function testGlobalRandomStateIsPreserved(testCase)
    previousState = rng(9182, 'twister');
    cleanup = onCleanup(@() rng(previousState));
    data = reshape((1:40) / 100, 20, 2);
    stateBeforeRun = rng;

    diss.run(data, deltaNormalConfig());

    verifyEqual(testCase, rng, stateBeforeRun);
end

function config = deltaNormalConfig()
    config = diss.config.defaults();
    config.model.kind = "deltaNormal";
end
