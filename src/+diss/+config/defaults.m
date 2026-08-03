function config = defaults()
%DEFAULTS Return the default configuration for the Diss pipeline.
%
% The defaults describe a full-sample Multi-GARCH run. Backtests require
% an explicit initialWindow so that the start of the out-of-sample period
% is never inferred silently.

config.schemaVersion = 2;

config.execution.mode = "full";

config.model.kind = "multiGarch";

config.dependence.kind = "dcc";
config.dependence.dynamic = "DCC";
config.dependence.archOrder = 1;
config.dependence.garchOrder = 1;
config.dependence.asymmetricOrder = 0;
config.dependence.copulaType = "gaussian";
config.dependence.families = strings(1, 0);
config.dependence.tails = "none";
config.dependence.options = struct();

config.marginal.enabled = true;
config.marginal.maxARLag = 1;
config.marginal.includeConstant = true;
config.marginal.archOrder = 1;
config.marginal.garchOrder = 1;
config.marginal.selectionPolicy = "initialWindow";
config.marginal.reestimateInBacktest = true;
config.marginal.engine = "econometricsToolbox";
config.marginal.candidateFamilies = ["garch", "egarch", "gjr"];
config.marginal.distribution = "Gaussian";
config.marginal.fixedSpecifications = struct([]);

config.backtest.windowType = "expanding";
config.backtest.initialWindow = [];
config.backtest.rollingWindow = [];
config.backtest.forecastHorizon = 1;
config.backtest.stepSize = 1;
config.backtest.marginalRefitEvery = 1;
config.backtest.includePartialFinalWindow = false;

config.simulation.numPaths = 10000;
config.simulation.seed = 1;

config.risk.probability = 0.99;
config.risk.portfolioWeights = [];
config.risk.returnAggregation = "linear";

config.inference.computeStandardErrors = false;

config.optimization.display = "off";
config.optimization.maxIterations = 500;
config.optimization.maxFunctionEvaluations = 10000;

config.data.missingAction = "error";

config.output.saveWindowModels = false;
config.output.saveSimulations = false;
config.output.saveTrainingData = false;

end
