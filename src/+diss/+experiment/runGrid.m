function results = runGrid(data, configurations)
%RUNGRID Run model specifications while reusing identical full-sample margins.

arguments
    data
    configurations {mustBeConfigurationCollection}
end

if isstruct(configurations)
    configurations = num2cell(configurations);
end
results = cell(size(configurations));
marginalStates = containers.Map('KeyType', 'char', 'ValueType', 'any');

for index = 1:numel(configurations)
    config = diss.config.validate(configurations{index}, data);
    canReuse = config.execution.mode == "full" && ...
        config.marginal.selectionPolicy ~= "eachWindow" && ...
        any(config.model.kind == ["univariateGarch", "multiGarch", ...
        "multiCopula"]);
    if canReuse
        key = char(diss.io.hashValue(config.marginal));
        if isKey(marginalStates, key)
            initialState = marginalStates(key);
        else
            initialState = struct();
        end
    else
        key = '';
        initialState = struct();
    end

    [results{index}, finalState] = diss.experiment.run( ...
        data, config, initialState);
    if canReuse && isfield(finalState, 'MarginalModels')
        marginalStates(key) = struct( ...
            'MarginalModels', {finalState.MarginalModels}, ...
            'MarginalSpecifications', ...
            {finalState.MarginalSpecifications});
    end
end

end

function mustBeConfigurationCollection(value)
if ~(iscell(value) || isstruct(value)) || isempty(value)
    error('diss:experiment:InvalidConfigurationCollection', ...
        'Configurations must be a nonempty cell or struct array.');
end
if iscell(value) && ~all(cellfun( ...
        @(item) isstruct(item) && isscalar(item), value(:)))
    error('diss:experiment:InvalidConfigurationCollection', ...
        'Every configuration must be a scalar struct.');
end
end
