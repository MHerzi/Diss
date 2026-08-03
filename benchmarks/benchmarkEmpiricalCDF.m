function results = benchmarkEmpiricalCDF
%BENCHMARKEMPIRICALCDF Compare legacy and vectorized PIT ranking.
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
    addpath(fullfile(repositoryRoot, 'legacy'));
    addpath(fullfile(repositoryRoot, 'src'));
    previousState = rng(3351, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    data = randn(5000, 20);

    legacyFunction = @() legacyEmpiricalCDF(data);
    modernFunction = @() empiricalCDF(data);
    expected = legacyFunction();
    actual = modernFunction();
    assert(isequal(actual, expected), ...
        'Modern and legacy empirical transforms differ.');

    legacySeconds = timeit(legacyFunction);
    modernSeconds = timeit(modernFunction);
    results = table(legacySeconds, modernSeconds, ...
        legacySeconds / modernSeconds, ...
        'VariableNames', {'LegacySeconds', 'ModernSeconds', 'Speedup'});
    disp(results)
end

function probabilities = legacyEmpiricalCDF(data)
    [observationCount, seriesCount] = size(data);
    probabilities = zeros(size(data));
    for series = 1:seriesCount
        indexedData = [data(:, series), (1:observationCount)'];
        sortedData = sortrows(indexedData, 1);
        rankedData = [sortedData, ...
            (1:observationCount)' / (observationCount + 1)];
        restoredData = sortrows(rankedData, 2);
        probabilities(:, series) = restoredData(:, 3);
    end
end
