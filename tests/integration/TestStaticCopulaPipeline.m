function tests = TestStaticCopulaPipeline
tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))));
addpath(repositoryRoot);
addpath(fullfile(repositoryRoot, 'src'));
end

function testInnovationCdfAndQuantileRoundTrip(testCase)
probabilities = linspace(0.01, 0.99, 99)';
distributions = {struct('Name', "Gaussian"), ...
    struct('Name', "t", 'DoF', 7)};
for index = 1:numel(distributions)
    distribution = distributions{index};
    values = diss.distributions.innovationQuantile( ...
        probabilities, distribution);
    reconstructed = diss.distributions.innovationCdf( ...
        values, distribution);
    verifyEqual(testCase, reconstructed, probabilities, ...
        'AbsTol', 2e-12);
end
end

function testGaussianCopulaRecoversSyntheticCorrelation(testCase)
previousState = rng(8803, 'twister');
cleanup = onCleanup(@() rng(previousState));
target = [1, 0.55, -0.20; 0.55, 1, 0.15; -0.20, 0.15, 1];
scores = randn(12000, 3) * chol(target);
pseudoObservations = 0.5 * erfc(-scores / sqrt(2));

copula = diss.dependence.copula.fitGaussian(pseudoObservations);

verifyEqual(testCase, copula.Correlation, target, 'AbsTol', 0.025);
verifyGreaterThan(testCase, copula.ExitFlag, 0);
verifyTrue(testCase, isfinite(copula.LogLikelihood));
end

function testPipelineIsReproducibleForFixedSeed(testCase)
assumeTrue(testCase, exist('arima', 'class') == 8);
assumeTrue(testCase, exist('fmincon', 'file') == 2);
previousState = rng(8810, 'twister');
cleanup = onCleanup(@() rng(previousState));
scores = randn(240, 2) * chol([1, 0.45; 0.45, 1]);
data = 0.015 * scores;
config = copulaConfig();

first = diss.runExperiment(data, config);
second = diss.runExperiment(data, config);

verifyEqual(testCase, first.Model.Dependence.Correlation, ...
    second.Model.Dependence.Correlation, 'AbsTol', 0);
verifyEqual(testCase, first.Forecast.ReturnQuantile, ...
    second.Forecast.ReturnQuantile, 'AbsTol', 0);
verifyTrue(testCase, isfinite(first.Forecast.ReturnQuantile));
verifyGreaterThan(testCase, ...
    first.Model.Dependence.Correlation(1, 2), 0.25);
end

function testDynamicCopulaFailsBeforeEstimation(testCase)
config = copulaConfig();
config.dependence.dynamic = "DCC";
verifyError(testCase, @() diss.prepare(randn(100, 2), config), ...
    'diss:config:UnsupportedCopulaSpecification');
end

function config = copulaConfig()
config = diss.config.defaults();
config.model.kind = "multiCopula";
config.dependence.kind = "copula";
config.dependence.dynamic = "none";
config.dependence.copulaType = "gaussian";
config.marginal.maxARLag = 0;
config.marginal.candidateFamilies = "garch";
config.simulation.numPaths = 4000;
config.simulation.seed = 771;
config.optimization.display = "off";
end
