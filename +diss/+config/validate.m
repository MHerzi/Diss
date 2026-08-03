function config = validate(config, data)
%VALIDATE Merge defaults, normalize values and validate a pipeline config.

arguments
    config (1, 1) struct
    data
end

config = mergeKnownFields(diss.config.defaults(), config, "config");
[observationCount, seriesCount] = dataSize(data);

config.execution.mode = canonicalText(config.execution.mode, ...
    ["full", "backtest"], "config.execution.mode");
config.model.kind = canonicalText(config.model.kind, [ ...
    "multiGarch", "multiCopula", "vineCopula", ...
    "multiMixCopula", "dVineMix", "histSimGaussian", ...
    "histSimCopula", "deltaNormal"], "config.model.kind");
config.model.dynamic = canonicalText(config.model.dynamic, ...
    ["none", "DCC", "ADCC", "GDCC", "AGDCC"], ...
    "config.model.dynamic");
config.model.copulaType = canonicalText(config.model.copulaType, ...
    ["gaussian", "t"], "config.model.copulaType");
config.model.tails = canonicalText(config.model.tails, ...
    ["none", "pareto", "empirical"], "config.model.tails");
config.model.families = normalizeFamilies(config.model.families);

validatePositiveInteger(config.schemaVersion, "config.schemaVersion");
validatePositiveInteger(config.model.dccP, "config.model.dccP");
validatePositiveInteger(config.model.dccQ, "config.model.dccQ");
validateNonnegativeInteger(config.model.dccG, "config.model.dccG");

config.marginal.enabled = logicalScalar(config.marginal.enabled, ...
    "config.marginal.enabled");
config.marginal.includeConstant = logicalScalar( ...
    config.marginal.includeConstant, "config.marginal.includeConstant");
config.marginal.reestimateInBacktest = logicalScalar( ...
    config.marginal.reestimateInBacktest, ...
    "config.marginal.reestimateInBacktest");
validateNonnegativeInteger(config.marginal.maxARLag, ...
    "config.marginal.maxARLag");
validateOrder(config.marginal.archOrder, seriesCount, ...
    "config.marginal.archOrder");
validateOrder(config.marginal.garchOrder, seriesCount, ...
    "config.marginal.garchOrder");
config.marginal.selectionPolicy = canonicalText( ...
    config.marginal.selectionPolicy, ...
    ["initialWindow", "eachWindow", "fixed"], ...
    "config.marginal.selectionPolicy");
config.marginal.engine = canonicalText(config.marginal.engine, ...
    "econometricsToolbox", "config.marginal.engine");
config.marginal.candidateFamilies = normalizeCandidateFamilies( ...
    config.marginal.candidateFamilies);
config.marginal.distribution = canonicalText( ...
    config.marginal.distribution, "Gaussian", ...
    "config.marginal.distribution");
if ~isstruct(config.marginal.fixedSpecifications)
    error('diss:config:InvalidFixedSpecifications', ...
        'config.marginal.fixedSpecifications must be a struct array.');
end
if config.marginal.selectionPolicy == "fixed" && ...
        numel(config.marginal.fixedSpecifications) ~= seriesCount
    error('diss:config:InvalidFixedSpecifications', ...
        ['Fixed marginal selection requires one specification per ', ...
        'return series (%d).'], seriesCount);
end

config.backtest.windowType = canonicalText(config.backtest.windowType, ...
    ["expanding", "rolling"], "config.backtest.windowType");
validatePositiveInteger(config.backtest.forecastHorizon, ...
    "config.backtest.forecastHorizon");
validatePositiveInteger(config.backtest.stepSize, ...
    "config.backtest.stepSize");
validatePositiveInteger(config.backtest.marginalRefitEvery, ...
    "config.backtest.marginalRefitEvery");
config.backtest.includePartialFinalWindow = logicalScalar( ...
    config.backtest.includePartialFinalWindow, ...
    "config.backtest.includePartialFinalWindow");

validatePositiveInteger(config.simulation.numPaths, ...
    "config.simulation.numPaths");
validateNonnegativeInteger(config.simulation.seed, ...
    "config.simulation.seed");
validateattributes(config.risk.probability, {'numeric'}, ...
    {'real', 'finite', 'scalar', '>', 0, '<', 1}, mfilename, ...
    "config.risk.probability");
config.risk.portfolioWeights = validatePortfolioWeights( ...
    config.risk.portfolioWeights, seriesCount);
config.risk.returnAggregation = canonicalText( ...
    config.risk.returnAggregation, "linear", ...
    "config.risk.returnAggregation");

config.inference.computeStandardErrors = logicalScalar( ...
    config.inference.computeStandardErrors, ...
    "config.inference.computeStandardErrors");
config.optimization.display = canonicalText(config.optimization.display, ...
    ["off", "final", "iter"], "config.optimization.display");
validatePositiveInteger(config.optimization.maxIterations, ...
    "config.optimization.maxIterations");
validatePositiveInteger(config.optimization.maxFunctionEvaluations, ...
    "config.optimization.maxFunctionEvaluations");
config.output.saveWindowModels = logicalScalar( ...
    config.output.saveWindowModels, "config.output.saveWindowModels");
config.output.saveSimulations = logicalScalar( ...
    config.output.saveSimulations, "config.output.saveSimulations");
config.data.missingAction = canonicalText(config.data.missingAction, ...
    ["error", "omitRows"], "config.data.missingAction");

multivariateKinds = ["multiGarch", "multiCopula", "vineCopula", ...
    "multiMixCopula", "dVineMix"];
if any(config.model.kind == multivariateKinds) && seriesCount < 2
    error('diss:config:TooFewSeries', ...
        '%s requires at least two return series.', config.model.kind);
end
if config.model.kind == "multiGarch"
    if config.model.dynamic == "none"
        error('diss:config:MissingDependenceDynamics', ...
            'Multi-GARCH requires a DCC dependence specification.');
    end
    if config.model.dccP ~= 1 || config.model.dccQ ~= 1
        error('diss:config:UnsupportedDccOrder', ...
            ['The package Multi-GARCH adapter currently requires ', ...
            'config.model.dccP = 1 and config.model.dccQ = 1.']);
    end
    if any(config.marginal.archOrder < 1) || ...
            any(config.marginal.garchOrder < 1)
        error('diss:config:InvalidMarginalOrder', ...
            ['The modern Multi-GARCH adapter requires positive ARCH ', ...
            'and GARCH orders.']);
    end
end

if config.execution.mode == "backtest"
    if isempty(config.backtest.initialWindow)
        error('diss:config:MissingInitialWindow', ...
            ['config.backtest.initialWindow must be set explicitly for ', ...
            'a backtest.']);
    end
    validatePositiveInteger(config.backtest.initialWindow, ...
        "config.backtest.initialWindow");
    if config.backtest.initialWindow >= observationCount
        error('diss:config:InvalidInitialWindow', ...
            ['config.backtest.initialWindow must be smaller than the ', ...
            'number of observations (%d).'], observationCount);
    end

    if config.backtest.windowType == "rolling"
        if isempty(config.backtest.rollingWindow)
            config.backtest.rollingWindow = ...
                config.backtest.initialWindow;
        end
        validatePositiveInteger(config.backtest.rollingWindow, ...
            "config.backtest.rollingWindow");
        if config.backtest.rollingWindow > config.backtest.initialWindow
            error('diss:config:InvalidRollingWindow', ...
                ['config.backtest.rollingWindow cannot exceed ', ...
                'config.backtest.initialWindow.']);
        end
    end
end

end

function merged = mergeKnownFields(defaults, supplied, path)
merged = defaults;
names = fieldnames(supplied);

for index = 1:numel(names)
    name = names{index};
    childPath = path + "." + name;
    if ~isfield(defaults, name)
        error('diss:config:UnknownField', ...
            'Unknown configuration field: %s.', childPath);
    end

    defaultValue = defaults.(name);
    suppliedValue = supplied.(name);
    if isstruct(defaultValue) && isstruct(suppliedValue)
        if isempty(fieldnames(defaultValue))
            merged.(name) = suppliedValue;
        else
            merged.(name) = mergeKnownFields( ...
                defaultValue, suppliedValue, childPath);
        end
    else
        merged.(name) = suppliedValue;
    end
end
end

function [observationCount, seriesCount] = dataSize(data)
if isnumeric(data)
    if ~ismatrix(data) || isempty(data)
        error('diss:data:InvalidShape', ...
            'Numeric input data must be a nonempty two-dimensional matrix.');
    end
    [observationCount, seriesCount] = size(data);
elseif istable(data) || istimetable(data)
    observationCount = height(data);
    seriesCount = width(data);
    if observationCount == 0 || seriesCount == 0
        error('diss:data:InvalidShape', ...
            'Table input data must contain observations and variables.');
    end
else
    error('diss:data:UnsupportedType', ...
        'Input data must be a numeric matrix, table or timetable.');
end
end

function value = canonicalText(value, candidates, name)
if ischar(value)
    value = string(value);
end
if ~isstring(value) || ~isscalar(value) || ismissing(value)
    error('diss:config:InvalidText', ...
        '%s must be a character vector or string scalar.', name);
end

match = find(strcmpi(strtrim(value), candidates), 1);
if isempty(match)
    error('diss:config:InvalidChoice', ...
        '%s must be one of: %s.', name, strjoin(candidates, ", "));
end
value = candidates(match);
end

function families = normalizeFamilies(families)
if isempty(families)
    families = strings(1, 0);
    return
end
if ischar(families)
    families = string({families});
elseif iscellstr(families) %#ok<ISCLSTR>
    families = string(families);
end
if ~isstring(families) || ~isvector(families) || any(ismissing(families))
    error('diss:config:InvalidFamilies', ...
        'config.model.families must be text values.');
end
families = reshape(lower(strtrim(families)), 1, []);
end

function families = normalizeCandidateFamilies(families)
families = normalizeFamilies(families);
if isempty(families)
    error('diss:config:MissingMarginalFamilies', ...
        'At least one marginal variance family must be configured.');
end
supported = ["garch", "egarch", "gjr"];
if any(~ismember(families, supported))
    invalid = families(find(~ismember(families, supported), 1));
    error('diss:config:UnsupportedMarginalFamily', ...
        'Unsupported marginal variance family: %s.', invalid);
end
families = unique(families, 'stable');
end

function value = logicalScalar(value, name)
if ~islogical(value) || ~isscalar(value)
    error('diss:config:InvalidLogical', ...
        '%s must be a logical scalar.', name);
end
end

function validatePositiveInteger(value, name)
validateattributes(value, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
    mfilename, name);
end

function validateNonnegativeInteger(value, name)
validateattributes(value, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'integer', 'nonnegative'}, ...
    mfilename, name);
end

function validateOrder(value, seriesCount, name)
validateattributes(value, {'numeric'}, ...
    {'real', 'finite', 'vector', 'integer', 'nonnegative', 'nonempty'}, ...
    mfilename, name);
if ~(isscalar(value) || numel(value) == seriesCount)
    error('diss:config:InvalidOrderLength', ...
        '%s must be scalar or contain one value per series (%d).', ...
        name, seriesCount);
end
end

function weights = validatePortfolioWeights(weights, seriesCount)
if isempty(weights)
    weights = ones(1, seriesCount) / seriesCount;
    return
end

validateattributes(weights, {'numeric'}, ...
    {'real', 'finite', 'vector', 'numel', seriesCount}, ...
    mfilename, "config.risk.portfolioWeights");
weights = reshape(double(weights), 1, []);
if abs(sum(weights) - 1) > 1e-10
    error('diss:config:InvalidPortfolioWeights', ...
        'config.risk.portfolioWeights must sum to one.');
end
end
