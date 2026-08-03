function config = fromLegacySpec(spec)
%FROMLEGACYSPEC Translate the current flat Spec struct into the new schema.
%
% This adapter is intentionally limited to fields used by MainFile.m. The
% returned configuration must still be checked with diss.config.validate.

arguments
    spec (1, 1) struct
end

config = diss.config.defaults();

if isfield(spec, 'purpose')
    config.execution.mode = string(spec.purpose);
end
if isfield(spec, 'ModelType')
    config.model.kind = mapModelKind(spec.ModelType);
end
if isfield(spec, 'DynamicType') && ~isempty(spec.DynamicType)
    config.model.dynamic = string(spec.DynamicType);
elseif isfield(spec, 'Dynamic') && ~isempty(spec.Dynamic)
    config.model.dynamic = string(spec.Dynamic);
end
if isfield(spec, 'dccP'), config.model.dccP = spec.dccP; end
if isfield(spec, 'dccQ'), config.model.dccQ = spec.dccQ; end
if isfield(spec, 'dccG'), config.model.dccG = spec.dccG; end
if isfield(spec, 'CopulaType')
    config.model.copulaType = mapCopulaType(spec.CopulaType);
end
if isfield(spec, 'family') && ~isempty(spec.family)
    config.model.families = string(spec.family);
end
if isfield(spec, 'tails') && ~isempty(spec.tails)
    config.model.tails = string(spec.tails);
end

if isfield(spec, 'univariate')
    config.marginal.enabled = legacyOnOff(spec.univariate, 'univariate');
end
if isfield(spec, 'arlag'), config.marginal.maxARLag = spec.arlag; end
if isfield(spec, 'const')
    config.marginal.includeConstant = logical(spec.const);
end
if isfield(spec, 'archP'), config.marginal.archOrder = spec.archP; end
if isfield(spec, 'garchQ'), config.marginal.garchOrder = spec.garchQ; end
if isfield(spec, 'uniBacktest')
    config.marginal.reestimateInBacktest = ...
        legacyOnOff(spec.uniBacktest, 'uniBacktest');
end

if isfield(spec, 'ForecastStart')
    config.backtest.initialWindow = spec.ForecastStart;
end
if isfield(spec, 'ForecastNumb')
    config.backtest.forecastHorizon = spec.ForecastNumb;
    config.backtest.stepSize = spec.ForecastNumb;
end
if isfield(spec, 'uniforecastP')
    config.backtest.marginalRefitEvery = spec.uniforecastP;
end

if isfield(spec, 'SimNumb')
    config.simulation.numPaths = spec.SimNumb;
end
if isfield(spec, 'beta')
    config.risk.probability = spec.beta;
end
if isfield(spec, 'stderrors')
    config.inference.computeStandardErrors = ...
        strcmpi(string(spec.stderrors), "an") || ...
        strcmpi(string(spec.stderrors), "on");
end

end

function kind = mapModelKind(value)
switch lower(strtrim(string(value)))
    case "multigarch"
        kind = "multiGarch";
    case "multicopula"
        kind = "multiCopula";
    case "vinecopula"
        kind = "vineCopula";
    case "multimixcopula"
        kind = "multiMixCopula";
    case "dvine-mix"
        kind = "dVineMix";
    case "histsim_gauss"
        kind = "histSimGaussian";
    case "histsim_copula"
        kind = "histSimCopula";
    case "deltanormal"
        kind = "deltaNormal";
    otherwise
        error('diss:config:UnknownLegacyModel', ...
            'Unknown legacy ModelType: %s.', string(value));
end
end

function type = mapCopulaType(value)
switch lower(strtrim(string(value)))
    case {"gauss", "gaussian"}
        type = "gaussian";
    case "t"
        type = "t";
    otherwise
        error('diss:config:UnknownLegacyCopula', ...
            'Unknown legacy CopulaType: %s.', string(value));
end
end

function value = legacyOnOff(input, fieldName)
input = lower(strtrim(string(input)));
if input == "on"
    value = true;
elseif input == "off"
    value = false;
else
    error('diss:config:InvalidLegacySwitch', ...
        'Legacy field %s must be ''on'' or ''off''.', fieldName);
end
end
