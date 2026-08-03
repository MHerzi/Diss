function results = benchmarkAnalyticMultiGarchVaR()
%BENCHMARKANALYTICMULTIGARCHVAR Compare analytic and Monte Carlo Gaussian VaR.

repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repositoryRoot, 'legacy'));
previousState = rng(2608, 'twister');
cleanup = onCleanup(@() rng(previousState));
forecastCount = 20;
seriesCount = 4;
simulationCount = 5000;
confidenceLevel = 0.99;
weights = ones(1, seriesCount) / seriesCount;
meanForecast = 0.001 * randn(forecastCount, seriesCount);
varianceForecast = 0.0001 + 0.0005 * rand(forecastCount, seriesCount);
baseCorrelation = toeplitz(0.4 .^ (0:seriesCount - 1));
correlationPath = repmat(baseCorrelation, 1, 1, forecastCount);

analyticFunction = @() analyticVaR(meanForecast, varianceForecast, ...
    correlationPath, weights, confidenceLevel);
monteCarloFunction = @() monteCarloVaR(meanForecast, varianceForecast, ...
    correlationPath, weights, confidenceLevel, simulationCount);

analyticResult = analyticFunction();
monteCarloResult = monteCarloFunction();
analyticSeconds = timeit(analyticFunction);
monteCarloSeconds = timeit(monteCarloFunction);
analyticStorage = whos('analyticResult');
estimatedMonteCarloPeakBytes = ...
    simulationCount * seriesCount * 8;

results = table(analyticSeconds, monteCarloSeconds, ...
    monteCarloSeconds / analyticSeconds, analyticStorage.bytes, ...
    estimatedMonteCarloPeakBytes, ...
    max(abs(analyticResult - monteCarloResult)), ...
    'VariableNames', {'AnalyticSeconds', 'MonteCarloSeconds', ...
    'Speedup', 'AnalyticResultBytes', 'EstimatedMonteCarloPeakBytes', ...
    'MaximumMonteCarloDifference'});
disp(results)

end

function quantileForecast = analyticVaR( ...
    means, variances, correlations, weights, confidenceLevel)
forecastCount = size(means, 1);
quantileForecast = zeros(forecastCount, 1);
normalQuantile = -sqrt(2) * erfcinv(2 * (1 - confidenceLevel));
for forecast = 1:forecastCount
    standardDeviations = sqrt(variances(forecast, :));
    covariance = correlations(:, :, forecast) .* ...
        (standardDeviations(:) * standardDeviations(:)');
    portfolioMean = means(forecast, :) * weights(:);
    portfolioStdDev = sqrt(weights * covariance * weights(:));
    quantileForecast(forecast) = ...
        portfolioMean + normalQuantile * portfolioStdDev;
end
end

function quantileForecast = monteCarloVaR( ...
    means, variances, correlations, weights, confidenceLevel, ...
    simulationCount)
forecastCount = size(means, 1);
quantileForecast = zeros(forecastCount, 1);
for forecast = 1:forecastCount
    simulations = simulateCorrelatedGaussianReturns( ...
        means(forecast, :), variances(forecast, :), ...
        correlations(:, :, forecast), simulationCount);
    portfolioSimulations = simulations * weights(:);
    quantileForecast(forecast) = quantile( ...
        portfolioSimulations, 1 - confidenceLevel);
end
end
