function evaluation = summarize(results)
%SUMMARIZE Produce common out-of-sample diagnostics for every model.

arguments
    results (1, 1) struct
end

evaluation = struct( ...
    'ObservationCount', 0, ...
    'ViolationCount', 0, ...
    'ViolationRate', NaN, ...
    'ExpectedViolationRate', NaN, ...
    'KupiecStatistic', NaN, ...
    'KupiecPValue', NaN);

if ~isfield(results, 'Backtest') || isempty(results.Backtest) || ...
        height(results.Backtest) == 0
    return
end

violations = logical(results.Backtest.Violation);
observationCount = numel(violations);
violationCount = sum(violations);
expectedRate = 1 - results.Config.risk.probability;
observedRate = violationCount / observationCount;

nullLogLikelihood = binomialLogLikelihood( ...
    violationCount, observationCount, expectedRate);
alternativeLogLikelihood = binomialLogLikelihood( ...
    violationCount, observationCount, observedRate);
statistic = max(0, -2 * (nullLogLikelihood - alternativeLogLikelihood));

evaluation.ObservationCount = observationCount;
evaluation.ViolationCount = violationCount;
evaluation.ViolationRate = observedRate;
evaluation.ExpectedViolationRate = expectedRate;
evaluation.KupiecStatistic = statistic;
evaluation.KupiecPValue = erfc(sqrt(statistic / 2));

end

function value = binomialLogLikelihood(successes, observations, probability)
failures = observations - successes;
value = weightedLog(successes, probability) + ...
    weightedLog(failures, 1 - probability);
end

function value = weightedLog(count, probability)
if count == 0
    value = 0;
elseif probability <= 0
    value = -Inf;
else
    value = count * log(probability);
end
end
