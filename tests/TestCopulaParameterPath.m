function tests = TestCopulaParameterPath
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
end

function testCalibrationMatchesOneClusterKMeansReference(testCase)
    previousState = rng(2918, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    data = rand(120, 5);
    processParameters = [0.2, 0.7, -0.15];

    for family = {'clayton', 'gumbel', 'rotclayton'}
        [expectedPath, expectedDistance] = legacyPath(family{1}, ...
            processParameters, data, 'kalibrieren', 0);
        [actualPath, actualDistance] = ...
            copulaParam_tv_Patton_grm2_var(family{1}, ...
            processParameters, 1, data, 'kalibrieren', 0);
        verifyEqual(testCase, actualPath, expectedPath, 'AbsTol', 1e-13);
        verifyEqual(testCase, actualDistance, expectedDistance, ...
            'AbsTol', 1e-14);
    end
    omittedHorizonPath = copulaParam_tv_Patton_grm2_var('clayton', ...
        processParameters, 1, data, 'kalibrieren');
    verifyEqual(testCase, omittedHorizonPath, ...
        legacyPath('clayton', processParameters, data, ...
        'kalibrieren', 0), 'AbsTol', 1e-13);
end

function testForecastMatchesOneClusterKMeansReference(testCase)
    previousState = rng(8923, 'twister');
    cleaner = onCleanup(@() rng(previousState));
    data = rand(80, 3);
    processParameters = [-0.1, 0.6, 0.2];
    horizon = 15;

    [expectedPath, expectedDistance] = legacyPath('clayton', ...
        processParameters, data, 'vorhersage', horizon);
    [actualPath, actualDistance] = copulaParam_tv_Patton_grm2_var( ...
        'clayton', processParameters, 1, data, 'vorhersage', horizon);

    verifyEqual(testCase, actualPath, expectedPath, 'AbsTol', 1e-13);
    verifyEqual(testCase, actualDistance, expectedDistance, ...
        'AbsTol', 1e-14);
end

function [parameterPath, generalizedDistance] = legacyPath( ...
    family, processParameters, data, operation, horizon)
    movingAverageLags = 10;
    observationCount = size(data, 1);
    if strcmp(operation, 'kalibrieren')
        horizon = 0;
    end
    data = [data; zeros(horizon, size(data, 2))];
    pathLength = observationCount + horizon;
    latentParameter = zeros(pathLength, 1);
    absoluteDistance = zeros(pathLength, 1);
    generalizedDistance = zeros(pathLength, 1);
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
        generalizedDistance(observation) = ...
            absoluteDistance(observation) / movingAverageLags;
        latentParameter(observation) = processParameters(1) + ...
            processParameters(2) * latentParameter(observation-1) + ...
            processParameters(3) * generalizedDistance(observation);
    end

    if strcmp(family, 'gumbel')
        parameterPath = exp(latentParameter) + 1 + 1e-6;
    else
        parameterPath = exp(latentParameter) + 1e-6;
    end
    parameterPath = min(parameterPath, 20);
    if strcmp(operation, 'vorhersage')
        parameterPath = parameterPath(observationCount + (1:horizon));
    end
end
