function adapter = resolveModel(modelName)
%RESOLVEMODEL Resolve a configured model to its common pipeline adapter.

arguments
    modelName (1, 1) string
end

persistent adapters
if isempty(adapters)
    adapters = containers.Map('KeyType', 'char', 'ValueType', 'any');
    deltaNormal = diss.models.deltaNormal.adapter();
    multiGarch = diss.models.multiGarch.adapter();
    univariateGarch = diss.models.univariateGarch.adapter();
    multiCopula = diss.models.multiCopula.adapter();
    adapters(char(deltaNormal.Name)) = deltaNormal;
    adapters(char(multiGarch.Name)) = multiGarch;
    adapters(char(univariateGarch.Name)) = univariateGarch;
    adapters(char(multiCopula.Name)) = multiCopula;
end

key = char(modelName);
if ~isKey(adapters, key)
    available = string(keys(adapters));
    error('diss:run:ModelNotImplemented', ...
        'Model %s is not registered. Available models: %s.', ...
        modelName, strjoin(sort(available), ", "));
end
adapter = adapters(key);

end
