function forecastOutput = forecast( ...
    model, forecastCount, config, observedUpdates)
%FORECAST Produce sequential one-step AR-GARCH-DCC risk forecasts.

arguments
    model (1, 1) struct
    forecastCount (1, 1) double {mustBeInteger, mustBePositive}
    config (1, 1) struct
    observedUpdates double {mustBeReal, mustBeFinite} = []
end

[marginalForecast, standardizedUpdates] = ...
    diss.marginal.forecastPath(model.Marginal, ...
    model.TrainingReturns, forecastCount, observedUpdates);
assetMean = marginalForecast.AssetMean;
assetVariance = marginalForecast.AssetVariance;
seriesCount = size(assetMean, 2);
correlationPath = zeros(seriesCount, seriesCount, forecastCount);
portfolioMean = zeros(forecastCount, 1);
portfolioStdDev = zeros(forecastCount, 1);
weights = config.risk.portfolioWeights;

alignedRows = model.Marginal.AlignmentOffset + 1: ...
    size(model.TrainingReturns, 1);
standardizedResiduals = model.Marginal.Residuals(alignedRows, :) ./ ...
    sqrt(model.Marginal.ConditionalVariances(alignedRows, :));
dependenceState = diss.dependence.initializeForecastState( ...
    model.Dependence, standardizedResiduals);

for step = 1:forecastCount
    correlation = dependenceState.Correlation;
    correlationPath(:, :, step) = correlation;
    standardDeviations = sqrt(max(assetVariance(step, :), 0));
    covariance = correlation .* ...
        (standardDeviations(:) * standardDeviations(:)');
    portfolioMean(step) = assetMean(step, :) * weights(:);
    portfolioVariance = weights * covariance * weights(:);
    portfolioStdDev(step) = sqrt(max(portfolioVariance, 0));
    if step < forecastCount
        dependenceState = diss.dependence.updateForecastState( ...
            dependenceState, standardizedUpdates(step, :));
    end
end

tailProbability = 1 - config.risk.probability;
standardNormalQuantile = -sqrt(2) * erfcinv(2 * tailProbability);
returnQuantile = portfolioMean + ...
    standardNormalQuantile * portfolioStdDev;

forecastOutput = struct();
forecastOutput.Count = forecastCount;
forecastOutput.ConfidenceLevel = config.risk.probability;
forecastOutput.AssetMean = assetMean;
forecastOutput.AssetVariance = assetVariance;
forecastOutput.Correlation = correlationPath;
forecastOutput.PortfolioMean = portfolioMean;
forecastOutput.PortfolioStdDev = portfolioStdDev;
forecastOutput.ReturnQuantile = returnQuantile;
forecastOutput.LossVaR = -returnQuantile;

end
