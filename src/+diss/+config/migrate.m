function config = migrate(config)
%MIGRATE Upgrade older flat model/dependence settings to schema version 2.

arguments
    config (1, 1) struct
end

if isfield(config, 'schemaVersion') && config.schemaVersion > 2
    error('diss:config:UnsupportedSchemaVersion', ...
        'Configuration schema version %d is newer than supported version 2.', ...
        config.schemaVersion);
end

if ~isfield(config, 'dependence') || ~isstruct(config.dependence)
    config.dependence = struct();
end

if isfield(config, 'model') && isstruct(config.model)
    [config, config.model] = moveField( ...
        config, config.model, 'dynamic', 'dynamic');
    [config, config.model] = moveField( ...
        config, config.model, 'dccP', 'archOrder');
    [config, config.model] = moveField( ...
        config, config.model, 'dccQ', 'garchOrder');
    [config, config.model] = moveField( ...
        config, config.model, 'dccG', 'asymmetricOrder');
    [config, config.model] = moveField( ...
        config, config.model, 'copulaType', 'copulaType');
    [config, config.model] = moveField( ...
        config, config.model, 'families', 'families');
    [config, config.model] = moveField( ...
        config, config.model, 'tails', 'tails');
    [config, config.model] = moveField( ...
        config, config.model, 'options', 'options');
end

if isfield(config, 'model') && isfield(config.model, 'kind') && ...
        ~isfield(config.dependence, 'kind')
    switch lower(string(config.model.kind))
        case "multigarch"
            config.dependence.kind = "dcc";
        case {"multicopula", "multimixcopula", "histsimcopula"}
            config.dependence.kind = "copula";
        case {"vinecopula", "dvinemix"}
            config.dependence.kind = "vine";
        case {"univariategarch", "deltanormal"}
            config.dependence.kind = "none";
        otherwise
            config.dependence.kind = "none";
    end
end
config.schemaVersion = 2;

end

function [config, model] = moveField(config, model, oldName, newName)
if isfield(model, oldName)
    config.dependence.(newName) = model.(oldName);
    model = rmfield(model, oldName);
end
end
