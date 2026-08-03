function forecastOutput = forecast( ...
    model, forecastCount, config, observedUpdates)
%FORECAST Simulate portfolio risk from a fitted static Gaussian copula.

arguments
    model (1, 1) struct
    forecastCount (1, 1) double {mustBeInteger, mustBePositive}
    config (1, 1) struct
    observedUpdates double {mustBeReal, mustBeFinite} = []
end

[marginalForecast, ~] = diss.marginal.forecastPath( ...
    model.Marginal, model.TrainingReturns, ...
    forecastCount, observedUpdates);
assetMean = marginalForecast.AssetMean;
assetVariance = marginalForecast.AssetVariance;
seriesCount = size(assetMean, 2);
simulationCount = config.simulation.numPaths;
weights = config.risk.portfolioWeights;
factor = chol(model.Dependence.Correlation, 'lower');
portfolioMean = assetMean * weights(:);
portfolioStdDev = zeros(forecastCount, 1);
returnQuantile = zeros(forecastCount, 1);
if config.output.saveSimulations
    simulations = cell(forecastCount, 1);
else
    simulations = {};
end

for step = 1:forecastCount
    gaussianScores = randn(simulationCount, seriesCount) * factor';
    uniforms = 0.5 * erfc(-gaussianScores / sqrt(2));
    uniforms = min(max(uniforms, realmin), 1 - eps);
    standardizedInnovations = zeros(simulationCount, seriesCount);
    for series = 1:seriesCount
        standardizedInnovations(:, series) = ...
            diss.distributions.innovationQuantile( ...
            uniforms(:, series), ...
            model.Marginal.Models{series}.Distribution);
    end
    scenarioReturns = assetMean(step, :) + ...
        standardizedInnovations .* sqrt(assetVariance(step, :));
    portfolioScenarios = scenarioReturns * weights(:);
    portfolioStdDev(step) = std(portfolioScenarios, 0);
    returnQuantile(step) = quantile( ...
        portfolioScenarios, 1 - config.risk.probability);
    if config.output.saveSimulations
        simulations{step} = scenarioReturns;
    end
end

forecastOutput = struct();
forecastOutput.Count = forecastCount;
forecastOutput.ConfidenceLevel = config.risk.probability;
forecastOutput.AssetMean = assetMean;
forecastOutput.AssetVariance = assetVariance;
forecastOutput.Correlation = repmat( ...
    model.Dependence.Correlation, 1, 1, forecastCount);
forecastOutput.PortfolioMean = portfolioMean;
forecastOutput.PortfolioStdDev = portfolioStdDev;
forecastOutput.ReturnQuantile = returnQuantile;
forecastOutput.LossVaR = -returnQuantile;
forecastOutput.Simulations = simulations;

end
