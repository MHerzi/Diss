function results = benchmarkCopulaParameterPath
%BENCHMARKCOPULAPARAMETERPATH Compare k-means and direct median paths.
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(repositoryRoot, 'legacy'));
    previousState = rng(19371, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    data = rand(600, 6);
    processParameters = [0.15, 0.72, -0.1];
    horizon = 50;

    legacyCall = @() legacyPath(processParameters, data, horizon);
    modernCall = @() copulaParam_tv_Patton_grm2_var('gumbel', ...
        processParameters, 1, data, 'vorhersage', horizon);
    legacySeconds = timeit(legacyCall);
    modernSeconds = timeit(modernCall);

    expected = legacyCall();
    actual = modernCall();
    assert(max(abs(actual - expected)) < 1e-12, ...
        'Modern and legacy parameter paths differ.');

    results = table(legacySeconds, modernSeconds, ...
        legacySeconds / modernSeconds, 'VariableNames', ...
        {'LegacySeconds', 'ModernSeconds', 'Speedup'});
    disp(results)
end

function parameterPath = legacyPath(processParameters, data, horizon)
    movingAverageLags = 10;
    observationCount = size(data, 1);
    data = [data; zeros(horizon, size(data, 2))];
    pathLength = observationCount + horizon;
    latentParameter = zeros(pathLength, 1);
    absoluteDistance = zeros(pathLength, 1);
    latentParameter(1) = processParameters(1);

    for observation = 2:pathLength
        if observation <= movingAverageLags
            [~, ~, absoluteDistance(observation)] = kmeans( ...
                data(1:movingAverageLags, :)', 1, ...
                'Distance', 'cityblock');
            absoluteDistance(1) = absoluteDistance(2);
        elseif observation > observationCount + 1
            absoluteDistance(observation) = mean(absoluteDistance( ...
                observation-movingAverageLags:observation-1));
        else
            [~, ~, absoluteDistance(observation)] = kmeans(data( ...
                observation-movingAverageLags:observation-1, :)', 1, ...
                'Distance', 'cityblock');
        end
        generalizedDistance = absoluteDistance(observation) / ...
            movingAverageLags;
        latentParameter(observation) = processParameters(1) + ...
            processParameters(2) * latentParameter(observation-1) + ...
            processParameters(3) * generalizedDistance;
    end

    parameterPath = min(exp(latentParameter) + 1 + 1e-6, 20);
    parameterPath = parameterPath(observationCount + (1:horizon));
end
