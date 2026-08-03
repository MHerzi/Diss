function forecast = forecastWindow( ...
    model, forecastCount, config, observedUpdates)
%FORECASTWINDOW Forecast one pipeline model through a common interface.

arguments
    model (1, 1) struct
    forecastCount (1, 1) double {mustBeInteger, mustBePositive}
    config (1, 1) struct
    observedUpdates double {mustBeReal, mustBeFinite} = []
end

if isfield(model, 'AdapterName')
    adapterName = model.AdapterName;
elseif isfield(model, 'Type')
    adapterName = model.Type;
else
    error('diss:run:MissingModelType', ...
        'The fitted model does not identify its adapter.');
end
adapter = diss.registry.resolveModel(adapterName);
forecast = adapter.Forecast( ...
    model, forecastCount, config, observedUpdates);

end
