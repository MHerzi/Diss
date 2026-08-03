function results = benchmarkDccCorrelationFilter
%BENCHMARKDCCCORRELATIONFILTER Compare old and modernized DCC objectives.
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
    addpath(fullfile(repositoryRoot, 'legacy'));
    addpath(fullfile(repositoryRoot, 'src'));

    previousState = rng(4129, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    observationCount = 1000;
    seriesCount = 8;
    stdResiduals = randn(observationCount, seriesCount);
    parameters = [0.10; 0.82];

    oldFunction = @() legacyDccObjective(parameters, stdResiduals);
    newFunction = @() dcc_mvgarch_likelihood_grm( ...
        parameters, stdResiduals, 1, 1, 1e-6);

    expected = oldFunction();
    actual = newFunction();
    assert(abs(expected - actual) <= 1e-9 * max(1, abs(expected)), ...
        'Modernized and legacy objectives are not numerically equivalent.');

    oldSeconds = timeit(oldFunction);
    newSeconds = timeit(newFunction);

    bytesPerDouble = 8;
    legacyStateBytes = 2 * seriesCount^2 * ...
        (observationCount + 1) * bytesPerDouble;
    modernStateBytes = seriesCount^2 * bytesPerDouble;

    results = table(oldSeconds, newSeconds, oldSeconds / newSeconds, ...
        legacyStateBytes, modernStateBytes, ...
        legacyStateBytes / modernStateBytes, ...
        'VariableNames', {'LegacySeconds', 'ModernSeconds', 'Speedup', ...
        'LegacyStateBytes', 'ModernStateBytes', 'StateMemoryReduction'});
    disp(results)
end

function logL = legacyDccObjective(parameters, stdResiduals)
    [observationCount, seriesCount] = size(stdResiduals);
    qBar = cov(stdResiduals);
    archParameter = parameters(1);
    garchParameter = parameters(2);
    lagCount = 1;

    Qt = zeros(seriesCount, seriesCount, observationCount + lagCount);
    Rt = zeros(seriesCount, seriesCount, observationCount + lagCount);
    Qt(:, :, 1) = qBar;
    paddedResiduals = [zeros(1, seriesCount); stdResiduals];
    likelihoods = zeros(observationCount, 1);
    qIntercept = qBar * ...
        (1 - archParameter^2 - garchParameter^2);

    for observation = 1:observationCount
        paddedIndex = observation + lagCount;
        laggedResidual = paddedResiduals(paddedIndex - 1, :);
        Qt(:, :, paddedIndex) = qIntercept + ...
            archParameter^2 * (laggedResidual' * laggedResidual) + ...
            garchParameter^2 * Qt(:, :, paddedIndex - 1);
        scale = sqrt(diag(Qt(:, :, paddedIndex)));
        correlation = Qt(:, :, paddedIndex) ./ (scale * scale');
        correlation(1:seriesCount + 1:end) = 1;
        Rt(:, :, paddedIndex) = correlation;
        residual = paddedResiduals(paddedIndex, :);
        likelihoods(observation) = log(det(correlation)) + ...
            residual * inv(correlation) * residual'; %#ok<MINV>
    end
    logL = sum(likelihoods) / 2;
end
