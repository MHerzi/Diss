function tests = TestMultiGarchPipeline
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))));
    addpath(repositoryRoot);
    addpath(fullfile(repositoryRoot, 'src'));
end

function testDccParameterMatricesHaveExpectedStructure(testCase)
    [arch, garchMatrices, asymmetric] = ...
        diss.dependence.parameterMatrices([0.1; 0.9], ...
        "DCC", 1, 1, 2);

    verifyEqual(testCase, arch(:, :, 1), 0.1 * eye(2));
    verifyEqual(testCase, garchMatrices(:, :, 1), 0.9 * eye(2));
    verifyEqual(testCase, size(asymmetric), [2, 2, 0]);

    [arch, garchMatrices, asymmetric] = ...
        diss.dependence.parameterMatrices( ...
        [0.1; 0.05; 0.9], "ADCC", 1, 1, 2);
    verifyEqual(testCase, arch(:, :, 1), 0.1 * eye(2));
    verifyEqual(testCase, asymmetric(:, :, 1), 0.05 * eye(2));
    verifyEqual(testCase, garchMatrices(:, :, 1), 0.9 * eye(2));

    [arch, garchMatrices, asymmetric] = ...
        diss.dependence.parameterMatrices( ...
        [0.1; 0.2; 0.8; 0.7], "GDCC", 1, 1, 2);
    verifyEqual(testCase, arch(:, :, 1), diag([0.1, 0.2]));
    verifyEqual(testCase, garchMatrices(:, :, 1), diag([0.8, 0.7]));
    verifyEqual(testCase, size(asymmetric), [2, 2, 0]);

    [arch, garchMatrices, asymmetric] = ...
        diss.dependence.parameterMatrices( ...
        [0.1; 0.2; 0.03; 0.04; 0.8; 0.7], ...
        "AGDCC", 1, 1, 2);
    verifyEqual(testCase, arch(:, :, 1), diag([0.1, 0.2]));
    verifyEqual(testCase, asymmetric(:, :, 1), diag([0.03, 0.04]));
    verifyEqual(testCase, garchMatrices(:, :, 1), diag([0.8, 0.7]));
end

function testDefaultCandidateSetUsesSupportedMatlabModels(testCase)
    config = diss.config.defaults();

    verifyEqual(testCase, config.marginal.engine, ...
        "econometricsToolbox");
    verifyEqual(testCase, config.marginal.candidateFamilies, ...
        ["garch", "egarch", "gjr"]);
    verifyEqual(testCase, config.marginal.distribution, "Gaussian");
end

function testLegacyMarginalFamilyIsRejected(testCase)
    config = diss.config.defaults();
    config.marginal.candidateFamilies = ["garch", "avgarch"];

    verifyError(testCase, @() diss.prepare(randn(100, 2), config), ...
        'diss:config:UnsupportedMarginalFamily');
end

function testUnsupportedDccOrdersFailDuringPreparation(testCase)
    config = diss.config.defaults();
    config.dependence.archOrder = 2;

    verifyError(testCase, @() diss.prepare(randn(100, 2), config), ...
        'diss:config:UnsupportedDccOrder');
end

function testFullModernMultiGarchPipelineProducesFiniteForecast(testCase)
    assumeTrue(testCase, exist('arima', 'class') == 8);
    previousState = rng(4521, 'twister');
    cleanup = onCleanup(@() rng(previousState));
    innovations = randn(220, 2);
    data = innovations * chol([0.0004, 0.00015; 0.00015, 0.0005]);
    config = minimalMultiGarchConfig();

    results = diss.run(data, config);

    verifyEqual(testCase, results.Model.Type, "multiGarch");
    verifyEqual(testCase, results.Model.Marginal.Engine, ...
        "econometricsToolbox");
    verifyClass(testCase, results.Model.Marginal.Models{1}, 'arima');
    verifyGreaterThan(testCase, ...
        results.Model.Dependence.ExitFlag, 0);
    verifyTrue(testCase, all(isfinite( ...
        results.Forecast.ReturnQuantile)));
    verifySize(testCase, results.Forecast.Correlation, [2, 2, 1]);
end

function testForecastDoesNotUseFutureRowsEarly(testCase)
    assumeTrue(testCase, exist('arima', 'class') == 8);
    previousState = rng(7742, 'twister');
    cleanup = onCleanup(@() rng(previousState));
    innovations = randn(220, 2);
    data = innovations * chol([0.0004, 0.00012; 0.00012, 0.0006]);
    config = minimalMultiGarchConfig();
    [model, ~] = diss.fitWindow(data, config);
    updatesA = [0.01, -0.01; 0.02, 0.01; -0.03, 0.02];
    updatesB = updatesA;
    updatesB(2, :) = [0.20, -0.20];

    forecastA = diss.forecastWindow(model, 3, config, updatesA);
    forecastB = diss.forecastWindow(model, 3, config, updatesB);

    verifyEqual(testCase, forecastA.ReturnQuantile(1:2), ...
        forecastB.ReturnQuantile(1:2), 'AbsTol', 1e-12);
    verifyGreaterThan(testCase, abs(forecastA.ReturnQuantile(3) - ...
        forecastB.ReturnQuantile(3)), 1e-12);
end

function testRecursiveForecastMatchesCompleteRefilterReference(testCase)
    assumeTrue(testCase, exist('arima', 'class') == 8);
    assumeTrue(testCase, exist('fmincon', 'file') == 2);
    previousState = rng(8831, 'twister');
    cleanup = onCleanup(@() rng(previousState));
    innovations = randn(220, 2);
    data = innovations * chol([0.0005, 0.00013; 0.00013, 0.0004]);
    config = minimalMultiGarchConfig();
    [model, ~] = diss.fitWindow(data, config);
    updates = 0.01 * randn(6, 2);

    actual = diss.forecastWindow(model, 6, config, updates);
    expected = completeRefilterForecast(model, 6, config, updates);

    verifyEqual(testCase, actual.AssetMean, expected.AssetMean, ...
        'AbsTol', 1e-10);
    verifyEqual(testCase, actual.AssetVariance, expected.AssetVariance, ...
        'AbsTol', 1e-10);
    verifyEqual(testCase, actual.Correlation, expected.Correlation, ...
        'AbsTol', 1e-10);
    verifyEqual(testCase, actual.ReturnQuantile, expected.ReturnQuantile, ...
        'AbsTol', 1e-10);
end

function forecastOutput = completeRefilterForecast( ...
    model, forecastCount, config, observedUpdates)
seriesCount = size(model.TrainingReturns, 2);
assetMean = zeros(forecastCount, seriesCount);
assetVariance = zeros(forecastCount, seriesCount);
correlationPath = zeros(seriesCount, seriesCount, forecastCount);
portfolioMean = zeros(forecastCount, 1);
portfolioStdDev = zeros(forecastCount, 1);
weights = config.risk.portfolioWeights;

for step = 1:forecastCount
    updatesBeforeForecast = observedUpdates(1:step - 1, :);
    history = [model.TrainingReturns; updatesBeforeForecast];
    [residuals, variances] = inferMarginalHistoryReference( ...
        model.Marginal.Models, history);
    for series = 1:seriesCount
        [assetMean(step, series), ~, assetVariance(step, series)] = ...
            diss.models.multiGarch.forecastMarginal( ...
            model.Marginal.Models{series}, history(:, series), ...
            residuals(:, series), variances(:, series));
    end
    alignedRows = model.Marginal.AlignmentOffset + 1:size(history, 1);
    standardizedResiduals = residuals(alignedRows, :) ./ ...
        sqrt(variances(alignedRows, :));
    correlation = diss.dependence.forecastCorrelation( ...
        model.Dependence, standardizedResiduals);
    correlationPath(:, :, step) = correlation;
    standardDeviations = sqrt(max(assetVariance(step, :), 0));
    covariance = correlation .* ...
        (standardDeviations(:) * standardDeviations(:)');
    portfolioMean(step) = assetMean(step, :) * weights(:);
    portfolioStdDev(step) = sqrt(max( ...
        weights * covariance * weights(:), 0));
end

normalQuantile = -sqrt(2) * erfcinv(2 * (1 - config.risk.probability));
returnQuantile = portfolioMean + normalQuantile * portfolioStdDev;
forecastOutput = struct( ...
    'AssetMean', assetMean, ...
    'AssetVariance', assetVariance, ...
    'Correlation', correlationPath, ...
    'ReturnQuantile', returnQuantile);
end

function [residuals, variances] = ...
    inferMarginalHistoryReference(models, history)
seriesCount = size(history, 2);
residuals = zeros(size(history));
variances = zeros(size(history));
for series = 1:seriesCount
    [residuals(:, series), variances(:, series)] = ...
        infer(models{series}, history(:, series));
end
end

function config = minimalMultiGarchConfig()
    config = diss.config.defaults();
    config.model.kind = "multiGarch";
    config.dependence.dynamic = "DCC";
    config.marginal.candidateFamilies = "garch";
    config.marginal.maxARLag = 0;
    config.marginal.archOrder = 1;
    config.marginal.garchOrder = 1;
    config.optimization.display = "off";
end
