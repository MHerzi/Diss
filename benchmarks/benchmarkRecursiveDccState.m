function results = benchmarkRecursiveDccState()
%BENCHMARKRECURSIVEDCCSTATE Compare recursive and complete-refilter DCC paths.

repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repositoryRoot, 'src'));
previousState = rng(3902, 'twister');
cleanup = onCleanup(@() rng(previousState));
trainingResiduals = randn(2000, 6);
updates = randn(40, 6);
dependence = struct( ...
    'Type', "DCC", ...
    'Parameters', [0.10; 0.80], ...
    'ArchOrder', 1, ...
    'GarchOrder', 1, ...
    'QBar', cov(trainingResiduals), ...
    'NegativeQBar', cov(min(trainingResiduals, 0)));

recursiveFunction = @() recursiveForecast( ...
    dependence, trainingResiduals, updates);
refilterFunction = @() completeRefilter( ...
    dependence, trainingResiduals, updates);
recursiveResult = recursiveFunction();
refilterResult = refilterFunction();
maximumDifference = max(abs(recursiveResult(:) - refilterResult(:)));
if maximumDifference > 1e-10
    error('diss:benchmark:RecursiveDccMismatch', ...
        'Recursive and complete-refilter paths differ by %.3g.', ...
        maximumDifference);
end

recursiveSeconds = timeit(recursiveFunction);
refilterSeconds = timeit(refilterFunction);
filteredState = diss.dependence.initializeForecastState( ...
    dependence, trainingResiduals);
assert(all(isfinite(filteredState.Correlation), 'all'));
stateInfo = whos('filteredState');
historyInfo = whos('trainingResiduals');
results = table(recursiveSeconds, refilterSeconds, ...
    refilterSeconds / recursiveSeconds, stateInfo.bytes, ...
    historyInfo.bytes, maximumDifference, ...
    'VariableNames', {'RecursiveSeconds', 'RefilterSeconds', ...
    'Speedup', 'RecursiveStateBytes', 'TrainingHistoryBytes', ...
    'MaximumDifference'});
disp(results)

end

function correlations = recursiveForecast(dependence, history, updates)
state = diss.dependence.initializeForecastState(dependence, history);
correlations = zeros(size(state.Correlation, 1), ...
    size(state.Correlation, 2), size(updates, 1) + 1);
correlations(:, :, 1) = state.Correlation;
for step = 1:size(updates, 1)
    state = diss.dependence.updateForecastState(state, updates(step, :));
    correlations(:, :, step + 1) = state.Correlation;
end
end

function correlations = completeRefilter(dependence, history, updates)
correlations = zeros(size(dependence.QBar, 1), ...
    size(dependence.QBar, 2), size(updates, 1) + 1);
correlations(:, :, 1) = diss.dependence.forecastCorrelation( ...
    dependence, history);
for step = 1:size(updates, 1)
    history = [history; updates(step, :)]; %#ok<AGROW>
    correlations(:, :, step + 1) = ...
        diss.dependence.forecastCorrelation(dependence, history);
end
end
