function tests = TestCopulaPdfModernization
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    repositoryRoot = fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))));
    addpath(repositoryRoot);
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
        fullfile(repositoryRoot, 'legacy')));
    addpath(fullfile(repositoryRoot, 'src'));
    testCase.TestData.Reference = load( ...
        fullfile(repositoryRoot, 'tests', 'legacyCopulaReference.mat'));
end

function testMatchesLegacyThreeDimensionalDensities(testCase)
    reference = testCase.TestData.Reference;
    actual = {
        copulapdfmultivariat_grm('gaussian', reference.u, reference.rho)
        copulapdfmultivariat_grm('t', reference.u, reference.rho, 7)
        copulapdfmultivariat_grm('clayton', reference.u, 1.5)
        copulapdfmultivariat_grm('gumbel', reference.u, 1.5)
        copulapdfmultivariat_grm('frank', reference.u, 4)
    };
    expected = {reference.gaussianLegacy; reference.tLegacy; ...
        reference.claytonLegacy; reference.gumbelLegacy; ...
        reference.frankLegacy};
    for familyIndex = 1:numel(actual)
        verifyEqual(testCase, actual{familyIndex}, expected{familyIndex}, ...
            'RelTol', 2e-12, 'AbsTol', 1e-13);
    end
end

function testBivariateDensitiesMatchMatlabImplementations(testCase)
    data = testCase.TestData.Reference.u(:, 1:2);
    rho = [1, 0.4; 0.4, 1];

    verifyEqual(testCase, ...
        copulapdfmultivariat_grm('gaussian', data, rho), ...
        copulapdf('Gaussian', data, rho), 'RelTol', 2e-12);
    verifyEqual(testCase, ...
        copulapdfmultivariat_grm('t', data, rho, 7), ...
        copulapdf('t', data, rho, 7), 'RelTol', 2e-12);
    verifyEqual(testCase, ...
        copulapdfmultivariat_grm('clayton', data, 1.5), ...
        copulapdf('Clayton', data, 1.5), 'RelTol', 2e-12);
    verifyEqual(testCase, ...
        copulapdfmultivariat_grm('gumbel', data, 1.5), ...
        copulapdf('Gumbel', data, 1.5), 'RelTol', 2e-12);
    verifyEqual(testCase, ...
        copulapdfmultivariat_grm('frank', data, 4), ...
        copulapdf('Frank', data, 4), 'RelTol', 2e-12);
end

function testRotatedClaytonUsesSurvivalTransformation(testCase)
    data = testCase.TestData.Reference.u(:, 1:2);
    actual = copulapdfmultivariat_grm('rotclayton', data, 1.5);
    expected = copulapdf('Clayton', 1 - data, 1.5);
    verifyEqual(testCase, actual, expected, 'RelTol', 2e-12);
end

function testIndependenceLimits(testCase)
    data = testCase.TestData.Reference.u;
    verifyEqual(testCase, ...
        copulapdfmultivariat_grm('gumbel', data, 1), ...
        ones(size(data, 1), 1), 'AbsTol', 2e-13);
    verifyEqual(testCase, ...
        copulapdfmultivariat_grm('frank', data, 1e-9), ...
        ones(size(data, 1), 1), 'AbsTol', 2e-13);
end

function testObservationSpecificParameters(testCase)
    data = testCase.TestData.Reference.u;
    theta = linspace(1.1, 2.5, size(data, 1))';
    density = copulapdfmultivariat_grm('gumbel', data, theta);
    verifySize(testCase, density, [size(data, 1), 1]);
    verifyTrue(testCase, all(isfinite(density) & density > 0));
end
