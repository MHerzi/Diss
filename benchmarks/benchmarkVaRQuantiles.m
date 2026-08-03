function results = benchmarkVaRQuantiles
%BENCHMARKVARQUANTILES Compare repeated and batched quantile extraction.
    previousState = rng(7712, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    simulatedReturns = randn(50000, 8);
    probabilities = [0.001, 0.01, 0.05, 0.10, 0.90, 0.95, 0.99, 0.999];

    oldFunction = @() repeatedSortedQuantiles( ...
        simulatedReturns, probabilities);
    newFunction = @() quantile(simulatedReturns, probabilities, 1);
    expected = oldFunction();
    actual = newFunction();
    assert(max(abs(expected(:) - actual(:))) <= 1e-14, ...
        'Batched and repeated quantiles differ.');

    oldSeconds = timeit(oldFunction);
    newSeconds = timeit(newFunction);
    results = table(oldSeconds, newSeconds, oldSeconds / newSeconds, ...
        'VariableNames', {'LegacySeconds', 'ModernSeconds', 'Speedup'});
    disp(results)
end

function values = repeatedSortedQuantiles(simulatedReturns, probabilities)
    sortedReturns = sort(simulatedReturns, 1);
    values = zeros(numel(probabilities), size(simulatedReturns, 2));
    for series = 1:size(simulatedReturns, 2)
        for probabilityIndex = 1:numel(probabilities)
            values(probabilityIndex, series) = quantile( ...
                sortedReturns(:, series), probabilities(probabilityIndex));
        end
    end
end
