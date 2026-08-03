function tests = TestUnivariateGarchPipeline
tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))));
addpath(repositoryRoot);
addpath(fullfile(repositoryRoot, 'src'));
end

function testGaussianGarchProducesFiniteForecast(testCase)
assumeTrue(testCase, exist('arima', 'class') == 8);
assumeTrue(testCase, exist('fmincon', 'file') == 2);
previousState = rng(5521, 'twister');
cleanup = onCleanup(@() rng(previousState));
data = 0.02 * randn(240, 1);
config = univariateConfig("Gaussian");

results = diss.runExperiment(data, config);

verifyEqual(testCase, results.Model.Type, "univariateGarch");
verifyEqual(testCase, results.Model.Marginal.Engine, ...
    "econometricsToolbox");
verifyTrue(testCase, all(isfinite(results.Forecast.ReturnQuantile)));
verifySize(testCase, results.Forecast.AssetVariance, [1, 1]);
end

function testStudentTQuantileIsStandardizedAndSymmetric(testCase)
distribution = struct('Name', "t", 'DoF', 8);
lower = diss.distributions.innovationQuantile(0.01, distribution);
upper = diss.distributions.innovationQuantile(0.99, distribution);
gaussian = diss.distributions.innovationQuantile( ...
    0.01, struct('Name', "Gaussian"));

verifyEqual(testCase, lower, -upper, 'AbsTol', 1e-12);
verifyLessThan(testCase, lower, gaussian);
verifyEqual(testCase, ...
    diss.distributions.innovationQuantile(0.5, distribution), 0);
end

function testUnivariateModelRejectsMultipleSeries(testCase)
config = univariateConfig("Gaussian");
verifyError(testCase, @() diss.prepare(randn(100, 2), config), ...
    'diss:config:UnivariateSeriesCount');
end

function testMultiGarchRejectsStudentTUntilJointRiskIsImplemented(testCase)
config = diss.config.defaults();
config.marginal.distribution = "t";
verifyError(testCase, @() diss.prepare(randn(100, 2), config), ...
    'diss:config:UnsupportedMultiGarchDistribution');
end

function config = univariateConfig(distribution)
config = diss.config.defaults();
config.model.kind = "univariateGarch";
config.marginal.maxARLag = 0;
config.marginal.candidateFamilies = "garch";
config.marginal.distribution = distribution;
config.optimization.display = "off";
end
