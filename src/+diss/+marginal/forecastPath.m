function [forecastOutput, standardizedUpdates] = forecastPath( ...
    marginal, trainingReturns, forecastCount, observedUpdates)
%FORECASTPATH Recursively forecast and update fitted AR-GARCH marginals.

arguments
    marginal (1, 1) struct
    trainingReturns double {mustBeReal, mustBeFinite, mustBeNonempty}
    forecastCount (1, 1) double {mustBeInteger, mustBePositive}
    observedUpdates double {mustBeReal, mustBeFinite} = []
end

[trainingCount, seriesCount] = size(trainingReturns);
if ~isempty(observedUpdates) && ...
        ~isequal(size(observedUpdates), [forecastCount, seriesCount])
    error('diss:marginal:InvalidObservedUpdates', ...
        ['Observed updates must contain one row per forecast and one ', ...
        'column per return series.']);
end

assetMean = zeros(forecastCount, seriesCount);
assetVariance = zeros(forecastCount, seriesCount);
standardizedUpdates = zeros(max(forecastCount - 1, 0), seriesCount);
historyCapacity = trainingCount + max(forecastCount - 1, 0);
returnHistory = zeros(historyCapacity, seriesCount);
residualHistory = zeros(historyCapacity, seriesCount);
varianceHistory = zeros(historyCapacity, seriesCount);
returnHistory(1:trainingCount, :) = trainingReturns;
residualHistory(1:trainingCount, :) = marginal.Residuals;
varianceHistory(1:trainingCount, :) = marginal.ConditionalVariances;

historyCount = trainingCount;
for step = 1:forecastCount
    for series = 1:seriesCount
        rows = presampleRows(historyCount, marginal.Specifications(series));
        [assetMean(step, series), ~, assetVariance(step, series)] = ...
            diss.marginal.forecastOneStep(marginal.Models{series}, ...
            returnHistory(rows, series), residualHistory(rows, series), ...
            varianceHistory(rows, series));
    end

    if step < forecastCount
        if isempty(observedUpdates)
            nextReturn = assetMean(step, :);
        else
            nextReturn = observedUpdates(step, :);
        end
        nextRow = historyCount + 1;
        returnHistory(nextRow, :) = nextReturn;
        for series = 1:seriesCount
            rows = presampleRows( ...
                historyCount, marginal.Specifications(series));
            [newResidual, newVariance] = infer( ...
                marginal.Models{series}, nextReturn(series), ...
                'Y0', returnHistory(rows, series), ...
                'E0', residualHistory(rows, series), ...
                'V0', varianceHistory(rows, series));
            residualHistory(nextRow, series) = newResidual(end);
            varianceHistory(nextRow, series) = newVariance(end);
        end
        if any(~isfinite(residualHistory(nextRow, :))) || ...
                any(~isfinite(varianceHistory(nextRow, :))) || ...
                any(varianceHistory(nextRow, :) <= 0)
            error('diss:marginal:InvalidRecursiveUpdate', ...
                'Recursive marginal filtering produced invalid values.');
        end
        standardizedUpdates(step, :) = residualHistory(nextRow, :) ./ ...
            sqrt(varianceHistory(nextRow, :));
        historyCount = nextRow;
    end
end

forecastOutput = struct( ...
    'AssetMean', assetMean, ...
    'AssetVariance', assetVariance);

end

function rows = presampleRows(historyCount, specification)
requiredCount = max([specification.ARLag, ...
    specification.ArchOrder, specification.GarchOrder, 1]);
firstRow = max(1, historyCount - requiredCount + 1);
rows = firstRow:historyCount;
end
