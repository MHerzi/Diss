function results = benchmarkDccForecastPath
%BENCHMARKDCCFORECASTPATH Compare full-path and ring-buffer forecasting.
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
    addpath(fullfile(repositoryRoot, 'legacy'));
    addpath(fullfile(repositoryRoot, 'src'));
    previousState = rng(6632, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    observationCount = 1500;
    seriesCount = 8;
    stdResiduals = randn(observationCount, seriesCount);
    qBar = cov(stdResiduals);
    identityMatrix = eye(seriesCount);
    archMatrices = 0.10 * identityMatrix;
    garchMatrices = 0.82 * identityMatrix;
    emptyMatrices = zeros(seriesCount, seriesCount, 0);

    legacyFunction = @() legacyForecast(stdResiduals, qBar, ...
        archMatrices, garchMatrices);
    modernFunction = @() dccCorrelationForecastPath(stdResiduals, qBar, ...
        archMatrices, garchMatrices, emptyMatrices, emptyMatrices, 1);
    expected = legacyFunction();
    actual = modernFunction();
    assert(max(abs(expected(:) - actual(:))) <= 1e-11, ...
        'Modern and legacy forecasts differ.');

    legacySeconds = timeit(legacyFunction);
    modernSeconds = timeit(modernFunction);
    bytesPerDouble = 8;
    legacyStateBytes = 2 * seriesCount^2 * ...
        (observationCount + 2) * bytesPerDouble;
    modernStateBytes = 3 * seriesCount^2 * bytesPerDouble;
    results = table(legacySeconds, modernSeconds, ...
        legacySeconds / modernSeconds, legacyStateBytes, ...
        modernStateBytes, legacyStateBytes / modernStateBytes, ...
        'VariableNames', {'LegacySeconds', 'ModernSeconds', 'Speedup', ...
        'LegacyStateBytes', 'ModernStateBytes', 'StateMemoryReduction'});
    disp(results)
end

function finalR = legacyForecast(stdResiduals, qBar, archMatrix, garchMatrix)
    [observationCount, seriesCount] = size(stdResiduals);
    stateCount = observationCount + 1;
    Qt = repmat(qBar, 1, 1, stateCount + 1);
    Rt = zeros(seriesCount, seriesCount, stateCount + 1);
    paddedResiduals = [zeros(1, seriesCount); stdResiduals];
    qIntercept = qBar - archMatrix' * qBar * archMatrix - ...
        garchMatrix' * qBar * garchMatrix;
    for state = 1:stateCount
        paddedState = state + 1;
        residual = paddedResiduals(paddedState - 1, :);
        Qt(:, :, paddedState) = qIntercept + ...
            archMatrix' * (residual' * residual) * archMatrix + ...
            garchMatrix' * Qt(:, :, paddedState - 1) * garchMatrix;
        scale = sqrt(diag(Qt(:, :, paddedState)));
        currentR = Qt(:, :, paddedState) ./ (scale * scale');
        currentR(1:seriesCount + 1:end) = 1;
        Rt(:, :, paddedState) = currentR;
    end
    finalR = Rt(:, :, end);
end
