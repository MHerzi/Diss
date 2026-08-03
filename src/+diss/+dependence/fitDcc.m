function [dependence, state] = fitDcc(stdResiduals, config, state)
%FITDCC Estimate DCC-family parameters with one modern optimizer setup.

arguments
    stdResiduals double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    state (1, 1) struct
end

if exist('fmincon', 'file') ~= 2
    error('diss:dependence:MissingOptimizationToolbox', ...
        'DCC estimation requires fmincon from Optimization Toolbox.');
end

[observationCount, seriesCount] = size(stdResiduals);
qBar = cov(stdResiduals);
qBar = (qBar + qBar') / 2;
[~, qBarStatus] = chol(qBar, 'lower');
if qBarStatus ~= 0
    error('diss:dependence:SingularResidualCovariance', ...
        ['The standardized-residual covariance must be positive ', ...
        'definite for DCC estimation.']);
end
negativeQBar = cov(min(stdResiduals, 0));
negativeQBar = (negativeQBar + negativeQBar') / 2;

dynamicType = config.dependence.dynamic;
archOrder = config.dependence.archOrder;
garchOrder = config.dependence.garchOrder;
parameterCount = dependenceParameterCount( ...
    dynamicType, archOrder, garchOrder, seriesCount);
defaultStart = defaultStartingValues( ...
    dynamicType, archOrder, garchOrder, seriesCount);
startingValues = previousStartingValues( ...
    state, dynamicType, parameterCount, defaultStart);
lowerBounds = repmat(1e-6, parameterCount, 1);
upperBounds = repmat(1 - 1e-6, parameterCount, 1);
constraintTolerance = 1e-7;

objective = @(parameters) dependenceObjective(parameters, ...
    stdResiduals, qBar, negativeQBar, dynamicType, ...
    archOrder, garchOrder);
nonlinearConstraint = @(parameters) positiveInterceptConstraint( ...
    parameters, qBar, negativeQBar, dynamicType, archOrder, ...
    garchOrder, constraintTolerance);
options = createOptimizationOptions(config, 'sqp');

[parameters, negativeLogLikelihood, exitFlag, optimizationOutput] = ...
    fmincon(objective, startingValues, [], [], [], [], ...
    lowerBounds, upperBounds, nonlinearConstraint, options);

if exitFlag <= 0 || ~isfinite(negativeLogLikelihood)
    retryOptions = createOptimizationOptions(config, 'interior-point');
    if any(~isfinite(parameters))
        retryStart = defaultStart;
    else
        retryStart = min(max(parameters, lowerBounds), upperBounds);
    end
    [retryParameters, retryLikelihood, retryExitFlag, retryOutput] = ...
        fmincon(objective, retryStart, [], [], [], [], ...
        lowerBounds, upperBounds, nonlinearConstraint, retryOptions);
    if retryExitFlag > exitFlag || retryLikelihood < negativeLogLikelihood
        parameters = retryParameters;
        negativeLogLikelihood = retryLikelihood;
        exitFlag = retryExitFlag;
        optimizationOutput = retryOutput;
    end
end

if exitFlag <= 0 || ~isfinite(negativeLogLikelihood)
    error('diss:dependence:OptimizationFailed', ...
        '%s estimation did not converge.', dynamicType);
end

[archMatrices, garchMatrices, asymmetricMatrices] = ...
    diss.dependence.parameterMatrices(parameters, dynamicType, ...
    archOrder, garchOrder, seriesCount);
[filteredLikelihood, isValid] = diss.dependence.correlation.filter( ...
    stdResiduals, [], qBar, archMatrices, garchMatrices, ...
    asymmetricMatrices, negativeQBar);
if ~isValid
    error('diss:dependence:InvalidFilteredCorrelation', ...
        'The fitted %s parameters produce an invalid correlation path.', ...
        dynamicType);
end

dependence = struct();
dependence.Type = dynamicType;
dependence.Parameters = parameters;
dependence.ArchOrder = archOrder;
dependence.GarchOrder = garchOrder;
dependence.QBar = qBar;
dependence.NegativeQBar = negativeQBar;
dependence.NegativeLogLikelihood = filteredLikelihood;
dependence.LogLikelihood = -filteredLikelihood;
dependence.AIC = 2 * parameterCount + 2 * filteredLikelihood;
dependence.BIC = parameterCount * log(observationCount) + ...
    2 * filteredLikelihood;
dependence.ExitFlag = exitFlag;
dependence.OptimizationOutput = optimizationOutput;

state.DependenceType = dynamicType;
state.DependenceParameters = parameters;

end

function value = dependenceObjective(parameters, stdResiduals, qBar, ...
    negativeQBar, dynamicType, archOrder, garchOrder)
penalty = 1e12;
try
    [archMatrices, garchMatrices, asymmetricMatrices] = ...
        diss.dependence.parameterMatrices(parameters, dynamicType, ...
        archOrder, garchOrder, size(stdResiduals, 2));
    [value, isValid] = diss.dependence.correlation.filter( ...
        stdResiduals, [], qBar, ...
        archMatrices, garchMatrices, asymmetricMatrices, negativeQBar);
    if ~isValid || ~isfinite(value)
        value = penalty;
    end
catch
    value = penalty;
end
end

function [constraint, equalityConstraint] = positiveInterceptConstraint( ...
    parameters, qBar, negativeQBar, dynamicType, archOrder, ...
    garchOrder, tolerance)
[archMatrices, garchMatrices, asymmetricMatrices] = ...
    diss.dependence.parameterMatrices(parameters, dynamicType, ...
    archOrder, garchOrder, size(qBar, 1));
intercept = qBar;
for lag = 1:size(archMatrices, 3)
    matrix = archMatrices(:, :, lag);
    intercept = intercept - matrix' * qBar * matrix;
end
for lag = 1:size(garchMatrices, 3)
    matrix = garchMatrices(:, :, lag);
    intercept = intercept - matrix' * qBar * matrix;
end
for lag = 1:size(asymmetricMatrices, 3)
    matrix = asymmetricMatrices(:, :, lag);
    intercept = intercept - matrix' * negativeQBar * matrix;
end
intercept = (intercept + intercept') / 2;
minimumEigenvalue = min(eig(intercept));
constraint = tolerance - minimumEigenvalue;
equalityConstraint = [];
end

function options = createOptimizationOptions(config, algorithm)
options = optimoptions('fmincon', ...
    'Algorithm', algorithm, ...
    'Display', char(config.optimization.display), ...
    'MaxIterations', config.optimization.maxIterations, ...
    'MaxFunctionEvaluations', ...
    config.optimization.maxFunctionEvaluations, ...
    'ConstraintTolerance', 1e-7, ...
    'OptimalityTolerance', 1e-5, ...
    'StepTolerance', 1e-8);
end

function count = dependenceParameterCount( ...
    dynamicType, archOrder, garchOrder, seriesCount)
switch dynamicType
    case "DCC"
        count = archOrder + garchOrder;
    case "ADCC"
        count = 2 * archOrder + garchOrder;
    case "GDCC"
        count = (archOrder + garchOrder) * seriesCount;
    case "AGDCC"
        count = (2 * archOrder + garchOrder) * seriesCount;
    otherwise
        error('diss:dependence:UnsupportedDynamicType', ...
            'Unsupported dynamic dependence type: %s.', dynamicType);
end
end

function values = defaultStartingValues( ...
    dynamicType, archOrder, garchOrder, seriesCount)
archValue = sqrt(0.03 / archOrder);
asymmetricValue = sqrt(0.01 / archOrder);
garchValue = sqrt(0.90 / garchOrder);
switch dynamicType
    case "DCC"
        values = [repmat(archValue, archOrder, 1); ...
            repmat(garchValue, garchOrder, 1)];
    case "ADCC"
        values = [repmat(archValue, archOrder, 1); ...
            repmat(asymmetricValue, archOrder, 1); ...
            repmat(garchValue, garchOrder, 1)];
    case "GDCC"
        values = [repmat(archValue, archOrder * seriesCount, 1); ...
            repmat(garchValue, garchOrder * seriesCount, 1)];
    case "AGDCC"
        values = [repmat(archValue, archOrder * seriesCount, 1); ...
            repmat(asymmetricValue, archOrder * seriesCount, 1); ...
            repmat(garchValue, garchOrder * seriesCount, 1)];
    otherwise
        error('diss:dependence:UnsupportedDynamicType', ...
            'Unsupported dynamic dependence type: %s.', dynamicType);
end
end

function values = previousStartingValues( ...
    state, dynamicType, parameterCount, defaultValues)
if isfield(state, 'DependenceType') && ...
        state.DependenceType == dynamicType && ...
        isfield(state, 'DependenceParameters') && ...
        numel(state.DependenceParameters) == parameterCount && ...
        all(isfinite(state.DependenceParameters))
    values = state.DependenceParameters(:);
else
    values = defaultValues;
end
end
