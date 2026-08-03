function tests = TestDataTransformHelpers
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
end

function testEmpiricalTransformMatchesLegacyRanking(testCase)
    data = [4, 2; 1, 8; 3, 5; 2, 1];
    expected = legacyEmpiricalCDF(data);
    verifyEqual(testCase, empiricalCDF(data), expected);
    verifyEqual(testCase, empiricalCDF_grm(data), expected);
end

function testEmpiricalCdfAtEvaluationPoints(testCase)
    data = [1, 5; 2, 4; 3, 3; 4, 2];
    points = [0, 3; 2.5, 5; 10, 1];
    expected = [0, 0.5; 0.5, 1; 1, 0];
    verifyEqual(testCase, empiricalCDF(data, points), expected);
end

function testRhoToThetaOrdering(testCase)
    rho = [1, .1, .2, .3; .1, 1, .4, .5; ...
        .2, .4, 1, .6; .3, .5, .6, 1];
    expected = [.1; .2; .3; .4; .5; .6];
    verifyEqual(testCase, rho2theta(rho), expected);
    verifyEqual(testCase, rho2theta_grm(rho), expected);
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
