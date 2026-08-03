function tests = TestVaRMultiGarch
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
end

function testPortfolioQuantilesComeFromPortfolioSimulations(testCase)
    previousState = rng(7481, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    forecastCount = 20;
    seriesCount = 3;
    simulationCount = 2500;
    meanForecast = 0.01 * randn(forecastCount, seriesCount);
    varianceForecast = 0.01 + 0.04 * rand(forecastCount, seriesCount);
    correlation = [1, 0.45, -0.10; 0.45, 1, 0.25; -0.10, 0.25, 1];
    correlationPath = repmat(correlation, 1, 1, forecastCount);
    realizedData = 0.02 * randn(forecastCount + 5, seriesCount);
    garchOutput = repmat({struct('dist', 'GAUSS')}, 1, seriesCount);
    spec = struct('ForecastNumb', forecastCount);

    [output, simulations] = VaR_MultiGARCH(meanForecast, ...
        varianceForecast, garchOutput, correlationPath, simulationCount, ...
        realizedData, 'off', spec);

    probabilities = [0.001, 0.01, 0.05, 0.10, 0.90, 0.95, 0.99, 0.999];
    expected = zeros(forecastCount, numel(probabilities));
    for forecastIndex = 1:forecastCount
        expected(forecastIndex, :) = quantile( ...
            mean(simulations{forecastIndex}, 2), probabilities, 1);
    end

    actual = [output.VaRPortfolio9991, output.VaRPortfolio991, ...
        output.VaRPortfolio951, output.VaRPortfolio901, ...
        output.VaRPortfolio101, output.VaRPortfolio51, ...
        output.VaRPortfolio11, output.VaRPortfolio0011];
    verifyEqual(testCase, actual, expected, 'AbsTol', 1e-14);
    verifyEqual(testCase, output.VaRPortfoli0011, ...
        output.VaRPortfolio0011);
end

function testUnsupportedMarginalDistributionFailsExplicitly(testCase)
    verifyError(testCase, @() VaR_MultiGARCH(zeros(10, 2), ones(10, 2), ...
        {struct('dist', 'GAUSS'), struct('dist', 'STUDENTST')}, ...
        repmat(eye(2), 1, 1, 10), 100, zeros(10, 2), 'off', struct()), ...
        'Diss:VaR:UnsupportedDistribution');
end
