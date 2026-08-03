function tests = TestExperimentArchitecture
% Tests for the project-level experiment architecture and compatibility API.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    repositoryRoot = fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))));
addpath(repositoryRoot);
addpath(fullfile(repositoryRoot, 'src'));
testCase.TestData.RepositoryRoot = repositoryRoot;
end

function testCompatibilityEntryPointMatchesCanonicalEngine(testCase)
data = reshape((1:60) / 1000, 30, 2);
config = deltaNormalConfig();

compatibility = diss.run(data, config);
canonical = diss.runExperiment(data, config);

verifyEqual(testCase, compatibility.Model, canonical.Model);
verifyEqual(testCase, compatibility.Forecast, canonical.Forecast);
verifyEqual(testCase, compatibility.Backtest, canonical.Backtest);
verifyEqual(testCase, compatibility.Config, canonical.Config);
verifyEqual(testCase, compatibility.Evaluation, canonical.Evaluation);
end

function testRegistryMatchesDirectAdapterCalls(testCase)
data = reshape((1:40) / 1000, 20, 2);
config = diss.config.validate(deltaNormalConfig(), data);
adapter = diss.registry.resolveModel("deltaNormal");

[directModel, directState] = adapter.Fit( ...
    data, config, struct(), struct());
[registeredModel, registeredState] = diss.fitWindow( ...
    data, config, struct(), struct());
directModel.AdapterName = adapter.Name;

verifyEqual(testCase, registeredModel, directModel);
verifyEqual(testCase, registeredState, directState);
verifyEqual(testCase, diss.registry.availableModels(), ...
    ["deltaNormal", "univariateGarch", "multiGarch", "multiCopula"]);
end

function testVersionOneConfigurationMigratesWithoutChangingMeaning(testCase)
oldConfig = struct();
oldConfig.schemaVersion = 1;
oldConfig.model = struct( ...
    'kind', "multiGarch", 'dynamic', "ADCC", ...
    'dccP', 1, 'dccQ', 1, 'dccG', 1, ...
    'copulaType', "gaussian", 'families', strings(1, 0), ...
    'tails', "none", 'options', struct());

resolved = diss.config.validate(oldConfig, randn(100, 2));

verifyEqual(testCase, resolved.schemaVersion, 2);
verifyEqual(testCase, resolved.dependence.kind, "dcc");
verifyEqual(testCase, resolved.dependence.dynamic, "ADCC");
verifyEqual(testCase, resolved.dependence.archOrder, 1);
verifyEqual(testCase, resolved.dependence.garchOrder, 1);
verifyEqual(testCase, resolved.dependence.asymmetricOrder, 1);
verifyFalse(testCase, isfield(resolved.model, 'dynamic'));
end

function testManifestHashesAreDeterministicAndDataSensitive(testCase)
data = reshape((1:40) / 1000, 20, 2);
first = diss.runExperiment(data, deltaNormalConfig());
second = diss.runExperiment(data, deltaNormalConfig());
changed = data;
changed(end) = changed(end) + 1e-8;
third = diss.runExperiment(changed, deltaNormalConfig());

verifyEqual(testCase, first.Manifest.ConfigHash, ...
    second.Manifest.ConfigHash);
verifyEqual(testCase, first.Manifest.DataHash, second.Manifest.DataHash);
verifyEqual(testCase, first.Manifest.ExperimentId, ...
    second.Manifest.ExperimentId);
verifyNotEqual(testCase, first.Manifest.DataHash, third.Manifest.DataHash);
verifyNotEqual(testCase, first.Manifest.ExperimentId, ...
    third.Manifest.ExperimentId);
end

function testResultArtifactRoundTrip(testCase)
data = reshape((1:40) / 1000, 20, 2);
results = diss.runExperiment(data, deltaNormalConfig());
outputRoot = string(tempname);
mkdir(outputRoot);
cleanup = onCleanup(@() rmdir(outputRoot, 's'));

runDirectory = diss.io.saveRun(results, outputRoot);
loaded = diss.io.loadRun(runDirectory);

verifyEqual(testCase, loaded.Model, results.Model);
verifyEqual(testCase, loaded.Forecast, results.Forecast);
verifyTrue(testCase, isfile(fullfile(runDirectory, 'manifest.json')));
end

function testBacktestWindowModelsAreCompactByDefault(testCase)
data = reshape((1:60) / 1000, 30, 2);
config = deltaNormalConfig();
config.execution.mode = "backtest";
config.backtest.initialWindow = 20;
config.backtest.forecastHorizon = 2;
config.backtest.stepSize = 2;
config.output.saveWindowModels = true;

results = diss.runExperiment(data, config);
storedModel = results.WindowModels{1};

verifyTrue(testCase, storedModel.IsCompact);
verifyFalse(testCase, isfield(storedModel, 'TrainingReturns'));
end

function testExperimentGridMatchesIndependentRuns(testCase)
data = reshape((1:60) / 1000, 30, 2);
firstConfig = deltaNormalConfig();
firstConfig.risk.probability = 0.95;
secondConfig = deltaNormalConfig();
secondConfig.risk.probability = 0.99;

grid = diss.experiment.runGrid(data, {firstConfig, secondConfig});
firstIndependent = diss.runExperiment(data, firstConfig);
secondIndependent = diss.runExperiment(data, secondConfig);

verifyEqual(testCase, grid{1}.Forecast, firstIndependent.Forecast);
verifyEqual(testCase, grid{2}.Forecast, secondIndependent.Forecast);
end

function testRecursiveDccStateMatchesCompleteRefilter(testCase)
previousState = rng(4601, 'twister');
cleanup = onCleanup(@() rng(previousState));
residuals = randn(150, 3);
dependence = syntheticDependence(residuals);
state = diss.dependence.initializeForecastState(dependence, residuals);
updates = randn(8, 3);
allResiduals = [residuals; updates];

verifyEqual(testCase, state.Correlation, ...
    diss.dependence.forecastCorrelation(dependence, residuals), ...
    'AbsTol', 1e-12);

for step = 1:8
    update = updates(step, :);
    state = diss.dependence.updateForecastState(state, update);
    expected = diss.dependence.forecastCorrelation( ...
        dependence, allResiduals(1:size(residuals, 1) + step, :));
    verifyEqual(testCase, state.Correlation, expected, ...
        'AbsTol', 1e-11);
end
end

function config = deltaNormalConfig()
config = diss.config.defaults();
config.model.kind = "deltaNormal";
end

function dependence = syntheticDependence(residuals)
dependence.Type = "ADCC";
dependence.Parameters = [0.10; 0.04; 0.80];
dependence.ArchOrder = 1;
dependence.GarchOrder = 1;
dependence.QBar = cov(residuals);
dependence.NegativeQBar = cov(min(residuals, 0));
end
