function forecastOutput = forecast( ...
    model, forecastCount, config, observedUpdates)
%FORECAST Forecast a univariate AR-GARCH conditional distribution.

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
distribution = model.Marginal.Models{1}.Distribution;
standardQuantile = diss.distributions.innovationQuantile( ...
    1 - config.risk.probability, distribution);
portfolioStdDev = sqrt(max(assetVariance, 0));
returnQuantile = assetMean + standardQuantile * portfolioStdDev;

forecastOutput = struct();
forecastOutput.Count = forecastCount;
forecastOutput.ConfidenceLevel = config.risk.probability;
forecastOutput.AssetMean = assetMean;
forecastOutput.AssetVariance = assetVariance;
forecastOutput.Correlation = ones(1, 1, forecastCount);
forecastOutput.PortfolioMean = assetMean;
forecastOutput.PortfolioStdDev = portfolioStdDev;
forecastOutput.ReturnQuantile = returnQuantile;
forecastOutput.LossVaR = -returnQuantile;

end
