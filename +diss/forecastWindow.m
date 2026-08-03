function forecast = forecastWindow( ...
    model, forecastCount, config, observedUpdates)
%FORECASTWINDOW Forecast one pipeline model through a common interface.

arguments
    model (1, 1) struct
    forecastCount (1, 1) double {mustBeInteger, mustBePositive}
    config (1, 1) struct
    observedUpdates double {mustBeReal, mustBeFinite} = []
end

switch model.Type
    case "deltaNormal"
        forecast = diss.models.deltaNormal.forecast( ...
            model, forecastCount, config);
    case "multiGarch"
        forecast = diss.models.multiGarch.forecast( ...
            model, forecastCount, config, observedUpdates);
    otherwise
        error('diss:run:ModelNotImplemented', ...
            'No forecast adapter is implemented for model %s.', model.Type);
end

end
