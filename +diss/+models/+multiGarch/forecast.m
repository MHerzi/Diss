function forecastOutput = forecast( ...
    model, forecastCount, config, observedUpdates)
%FORECAST Produce sequential one-step AR-GARCH-DCC risk forecasts.
%
% For a backtest block, row s of observedUpdates is incorporated only after
% forecast s has been produced. This prevents look-ahead while allowing
% filters to update between parameter re-estimations.

arguments
    model (1, 1) struct
    forecastCount (1, 1) double {mustBeInteger, mustBePositive}
    config (1, 1) struct
    observedUpdates double {mustBeReal, mustBeFinite} = []
end

seriesCount = size(model.TrainingReturns, 2);
if ~isempty(observedUpdates) && ...
        ~isequal(size(observedUpdates), [forecastCount, seriesCount])
    error('diss:multiGarch:InvalidObservedUpdates', ...
        ['Observed backtest updates must contain one row per forecast ', ...
        'and one column per return series.']);
end

assetMean = zeros(forecastCount, seriesCount);
assetVariance = zeros(forecastCount, seriesCount);
correlationPath = zeros(seriesCount, seriesCount, forecastCount);
portfolioMean = zeros(forecastCount, 1);
portfolioStdDev = zeros(forecastCount, 1);
recursiveUpdates = zeros(0, seriesCount);
weights = config.risk.portfolioWeights;

for step = 1:forecastCount
    if isempty(observedUpdates)
        updatesBeforeForecast = recursiveUpdates;
    else
        updatesBeforeForecast = observedUpdates(1:step - 1, :);
    end
    history = [model.TrainingReturns; updatesBeforeForecast];
    [residuals, variances] = inferMarginalHistory( ...
        model.Marginal.Models, history);

    for series = 1:seriesCount
        [assetMean(step, series), ~, assetVariance(step, series)] = ...
            diss.models.multiGarch.forecastMarginal( ...
            model.Marginal.Models{series}, history(:, series), ...
            residuals(:, series), variances(:, series));
    end

    alignedRows = model.Marginal.AlignmentOffset + 1:size(history, 1);
    stdResiduals = residuals(alignedRows, :) ./ ...
        sqrt(variances(alignedRows, :));
    correlation = diss.dependence.forecastCorrelation( ...
        model.Dependence, stdResiduals);
    correlationPath(:, :, step) = correlation;

    standardDeviations = sqrt(max(assetVariance(step, :), 0));
    covariance = correlation .* ...
        (standardDeviations(:) * standardDeviations(:)');
    portfolioMean(step) = assetMean(step, :) * weights(:);
    portfolioVariance = weights * covariance * weights(:);
    portfolioStdDev(step) = sqrt(max(portfolioVariance, 0));

    if isempty(observedUpdates)
        recursiveUpdates(end + 1, :) = assetMean(step, :); %#ok<AGROW>
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

function [residuals, variances] = inferMarginalHistory(models, history)
seriesCount = size(history, 2);
residuals = zeros(size(history));
variances = zeros(size(history));
for series = 1:seriesCount
    [residuals(:, series), variances(:, series)] = ...
        infer(models{series}, history(:, series));
end
if any(~isfinite(residuals(:))) || any(~isfinite(variances(:))) || ...
        any(variances(:) <= 0)
    error('diss:multiGarch:InvalidMarginalInference', ...
        'Marginal filtering produced invalid residuals or variances.');
end
end
