function [marginal, state] = fit( ...
    returns, config, state, refitMarginals)
%FIT Fit AR-GARCH marginals with Econometrics Toolbox objects.

arguments
    returns double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    state (1, 1) struct
    refitMarginals (1, 1) logical
end

assertEconometricsToolboxAvailable();
seriesCount = size(returns, 2);
hasPreviousModels = isfield(state, 'MarginalModels') && ...
    numel(state.MarginalModels) == seriesCount;
hasPreviousSpecifications = isfield(state, 'MarginalSpecifications') && ...
    numel(state.MarginalSpecifications) == seriesCount;
if config.marginal.selectionPolicy == "eachWindow"
    refitMarginals = true;
end

if config.marginal.selectionPolicy == "fixed"
    specifications = normalizeFixedSpecifications( ...
        config.marginal.fixedSpecifications, config, seriesCount);
    selectSpecifications = false;
elseif config.marginal.selectionPolicy == "eachWindow"
    specifications = struct([]);
    selectSpecifications = true;
elseif hasPreviousSpecifications
    specifications = state.MarginalSpecifications;
    selectSpecifications = false;
else
    specifications = struct([]);
    selectSpecifications = true;
end

if ~refitMarginals && hasPreviousModels && hasPreviousSpecifications
    models = state.MarginalModels;
    specifications = state.MarginalSpecifications;
    logLikelihood = nan(1, seriesCount);
    bic = nan(1, seriesCount);
    exitFlag = nan(1, seriesCount);
    parameterCovariances = cell(1, 0);
else
    models = cell(1, seriesCount);
    fittedSpecifications = repmat(emptySpecification(), 1, seriesCount);
    logLikelihood = zeros(1, seriesCount);
    bic = zeros(1, seriesCount);
    exitFlag = zeros(1, seriesCount);
    if config.inference.computeStandardErrors
        parameterCovariances = cell(1, seriesCount);
    else
        parameterCovariances = cell(1, 0);
    end

    for series = 1:seriesCount
        if selectSpecifications
            [models{series}, fittedSpecifications(series), ...
                logLikelihood(series), bic(series), exitFlag(series), ...
                parameterCovariance] = selectAndFitSeries( ...
                returns(:, series), config, series);
        else
            [models{series}, logLikelihood(series), bic(series), ...
                exitFlag(series), parameterCovariance] = fitSpecification( ...
                returns(:, series), specifications(series), config);
            fittedSpecifications(series) = specifications(series);
        end
        if config.inference.computeStandardErrors
            parameterCovariances{series} = parameterCovariance;
        end
    end
    specifications = fittedSpecifications;
end

[residuals, conditionalVariances, inferredLogLikelihood] = ...
    inferMarginals(models, returns);
missingLikelihood = isnan(logLikelihood);
logLikelihood(missingLikelihood) = inferredLogLikelihood(missingLikelihood);

alignmentOffset = max([[specifications.ARLag], ...
    [specifications.ArchOrder], [specifications.GarchOrder]]);
if alignmentOffset >= size(returns, 1) - 1
    error('diss:multiGarch:InsufficientAlignedData', ...
        'Too few observations remain after marginal-model alignment.');
end
alignedRows = alignmentOffset + 1:size(returns, 1);
alignedVariances = conditionalVariances(alignedRows, :);
alignedResiduals = residuals(alignedRows, :);
if any(alignedVariances(:) <= 0) || ...
        any(~isfinite(alignedVariances(:))) || ...
        any(~isfinite(alignedResiduals(:)))
    error('diss:multiGarch:InvalidMarginalInference', ...
        'Marginal inference returned invalid residuals or variances.');
end

marginal = struct();
marginal.Engine = "econometricsToolbox";
marginal.Models = models;
marginal.Specifications = specifications;
marginal.LogLikelihood = logLikelihood;
marginal.BIC = bic;
marginal.ExitFlag = exitFlag;
marginal.Residuals = residuals;
marginal.ConditionalVariances = conditionalVariances;
marginal.StandardizedResiduals = ...
    alignedResiduals ./ sqrt(alignedVariances);
marginal.AlignmentOffset = alignmentOffset;
marginal.ParameterCovariances = parameterCovariances;

state.MarginalModels = models;
state.MarginalSpecifications = specifications;

end

function [bestModel, bestSpecification, bestLogLikelihood, bestBic, ...
    bestExitFlag, bestParameterCovariance] = ...
    selectAndFitSeries(series, config, seriesIndex)

bestModel = [];
bestSpecification = emptySpecification();
bestLogLikelihood = -Inf;
bestBic = Inf;
bestExitFlag = NaN;
bestParameterCovariance = [];
failureMessages = strings(0, 1);
archOrder = orderForSeries(config.marginal.archOrder, seriesIndex);
garchOrder = orderForSeries(config.marginal.garchOrder, seriesIndex);

for family = config.marginal.candidateFamilies
    for arLag = 0:config.marginal.maxARLag
        specification = struct( ...
            'Family', family, ...
            'ARLag', arLag, ...
            'ArchOrder', archOrder, ...
            'GarchOrder', garchOrder, ...
            'Distribution', config.marginal.distribution);
        try
            [model, logLikelihood, currentBic, currentExitFlag, ...
                parameterCovariance] = fitSpecification( ...
                series, specification, config);
            if currentBic < bestBic
                bestModel = model;
                bestSpecification = specification;
                bestLogLikelihood = logLikelihood;
                bestBic = currentBic;
                bestExitFlag = currentExitFlag;
                bestParameterCovariance = parameterCovariance;
            end
        catch exception
            failureMessages(end + 1, 1) = family + " AR(" + ...
                arLag + "): " + string(exception.message); %#ok<AGROW>
        end
    end
end

if isempty(bestModel)
    if isempty(failureMessages)
        details = "No candidate specification was generated.";
    else
        details = strjoin(failureMessages, newline);
    end
    error('diss:multiGarch:MarginalSelectionFailed', ...
        'All marginal candidates failed for series %d.%s%s', ...
        seriesIndex, newline, details);
end
end

function [estimatedModel, logLikelihood, bic, exitFlag, ...
    parameterCovariance] = fitSpecification(series, specification, config)

varianceModel = createVarianceModel(specification);
if specification.ARLag == 0
    arLags = [];
else
    arLags = 1:specification.ARLag;
end
if config.marginal.includeConstant
    constant = NaN;
else
    constant = 0;
end
template = arima('Constant', constant, 'ARLags', arLags, ...
    'Variance', varianceModel, ...
    'Distribution', char(specification.Distribution));
options = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ...
    'Display', 'off', ...
    'MaxIterations', config.optimization.maxIterations, ...
    'MaxFunctionEvaluations', ...
    config.optimization.maxFunctionEvaluations);

[estimatedModel, parameterCovariance, logLikelihood, information] = ...
    estimate(template, series, 'Display', 'off', 'Options', options);
exitFlag = information.exitflag;
if isempty(exitFlag) || exitFlag <= 0 || ~isfinite(logLikelihood)
    error('diss:multiGarch:MarginalOptimizationFailed', ...
        'Marginal optimization did not converge.');
end

parameterCount = countEstimatedParameters(specification, ...
    config.marginal.includeConstant);
bic = -2 * logLikelihood + parameterCount * log(numel(series));
end

function varianceModel = createVarianceModel(specification)
switch specification.Family
    case "garch"
        varianceModel = garch( ...
            specification.GarchOrder, specification.ArchOrder);
    case "egarch"
        varianceModel = egarch( ...
            specification.GarchOrder, specification.ArchOrder);
    case "gjr"
        varianceModel = gjr( ...
            specification.GarchOrder, specification.ArchOrder);
    otherwise
        error('diss:multiGarch:UnsupportedMarginalFamily', ...
            'Unsupported marginal family: %s.', specification.Family);
end
end

function count = countEstimatedParameters(specification, includeConstant)
meanParameterCount = specification.ARLag + double(includeConstant);
switch specification.Family
    case "garch"
        varianceParameterCount = 1 + specification.GarchOrder + ...
            specification.ArchOrder;
    case {"egarch", "gjr"}
        varianceParameterCount = 1 + specification.GarchOrder + ...
            2 * specification.ArchOrder;
    otherwise
        error('diss:multiGarch:UnsupportedMarginalFamily', ...
            'Unsupported marginal family: %s.', specification.Family);
end
distributionParameterCount = double( ...
    strcmpi(specification.Distribution, "t"));
count = meanParameterCount + varianceParameterCount + ...
    distributionParameterCount;
end

function [residuals, variances, logLikelihood] = ...
    inferMarginals(models, returns)
seriesCount = size(returns, 2);
residuals = zeros(size(returns));
variances = zeros(size(returns));
logLikelihood = zeros(1, seriesCount);
for series = 1:seriesCount
    [residuals(:, series), variances(:, series), ...
        logLikelihood(series)] = infer(models{series}, returns(:, series));
end
end

function specifications = normalizeFixedSpecifications( ...
    specifications, config, seriesCount)
requiredFields = {'Family', 'ARLag', 'ArchOrder', 'GarchOrder'};
for series = 1:seriesCount
    for field = requiredFields
        if ~isfield(specifications(series), field{1})
            error('diss:multiGarch:InvalidFixedSpecification', ...
                'Fixed specification %d is missing field %s.', ...
                series, field{1});
        end
    end
    family = lower(string(specifications(series).Family));
    if ~ismember(family, config.marginal.candidateFamilies)
        error('diss:multiGarch:InvalidFixedSpecification', ...
            'Fixed specification %d uses disabled family %s.', ...
            series, family);
    end
    specifications(series).Family = family;
    validateattributes(specifications(series).ARLag, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'});
    validateattributes(specifications(series).ArchOrder, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    validateattributes(specifications(series).GarchOrder, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    specifications(series).Distribution = ...
        config.marginal.distribution;
end
end

function specification = emptySpecification()
specification = struct( ...
    'Family', "", ...
    'ARLag', 0, ...
    'ArchOrder', 1, ...
    'GarchOrder', 1, ...
    'Distribution', "Gaussian");
end

function order = orderForSeries(orders, series)
if isscalar(orders)
    order = orders;
else
    order = orders(series);
end
end

function assertEconometricsToolboxAvailable()
requiredClasses = {'arima', 'garch', 'egarch', 'gjr'};
missing = requiredClasses(cellfun( ...
    @(name) exist(name, 'class') ~= 8, requiredClasses));
if ~isempty(missing)
    error('diss:multiGarch:MissingEconometricsToolbox', ...
        'Missing Econometrics Toolbox classes: %s.', strjoin(missing, ', '));
end
end
