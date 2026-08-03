function forecast = forecast(model, forecastCount, config, observedUpdates)
%FORECAST Return repeated one-step Delta-Normal portfolio risk forecasts.

arguments
    model (1, 1) struct
    forecastCount (1, 1) double {mustBeInteger, mustBePositive}
    config (1, 1) struct
    observedUpdates double {mustBeReal, mustBeFinite} = []
end

% Delta-Normal parameters are unconditional; observed updates are ignored.
if ~isempty(observedUpdates) && size(observedUpdates, 1) ~= forecastCount
    error('diss:deltaNormal:InvalidObservedUpdates', ...
        'Observed updates must contain one row per forecast.');
end

tailProbability = 1 - config.risk.probability;
standardNormalQuantile = -sqrt(2) * erfcinv(2 * tailProbability);
returnQuantile = model.PortfolioMean + ...
    standardNormalQuantile * model.PortfolioStdDev;

forecast = struct();
forecast.Count = forecastCount;
forecast.ConfidenceLevel = config.risk.probability;
forecast.PortfolioMean = repmat(model.PortfolioMean, forecastCount, 1);
forecast.PortfolioStdDev = repmat( ...
    model.PortfolioStdDev, forecastCount, 1);
forecast.ReturnQuantile = repmat(returnQuantile, forecastCount, 1);
forecast.LossVaR = -forecast.ReturnQuantile;

end
