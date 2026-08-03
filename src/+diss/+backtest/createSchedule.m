function schedule = createSchedule(observationCount, options)
%CREATESCHEDULE Create one shared schedule for all backtest model types.

arguments
    observationCount (1, 1) double {mustBeInteger, mustBePositive}
    options (1, 1) struct
end

initialWindow = options.initialWindow;
forecastHorizon = options.forecastHorizon;
stepSize = options.stepSize;

forecastStart = (initialWindow + 1:stepSize:observationCount)';
forecastEnd = forecastStart + forecastHorizon - 1;

if options.includePartialFinalWindow
    forecastEnd = min(forecastEnd, observationCount);
else
    completeWindow = forecastEnd <= observationCount;
    forecastStart = forecastStart(completeWindow);
    forecastEnd = forecastEnd(completeWindow);
end

if isempty(forecastStart)
    error('diss:backtest:NoForecastWindows', ...
        ['The backtest settings do not create a complete forecast ', ...
        'window for %d observations.'], observationCount);
end

estimationEnd = forecastStart - 1;
switch options.windowType
    case "expanding"
        estimationStart = ones(size(estimationEnd));
    case "rolling"
        estimationStart = estimationEnd - options.rollingWindow + 1;
    otherwise
        error('diss:backtest:InvalidWindowType', ...
            'Unsupported window type: %s.', options.windowType);
end

windowCount = numel(forecastStart);
window = (1:windowCount)';
estimationLength = estimationEnd - estimationStart + 1;
forecastLength = forecastEnd - forecastStart + 1;
refitMarginal = false(windowCount, 1);
lastMarginalFitEnd = -Inf;
for index = 1:windowCount
    if index == 1 || ...
            estimationEnd(index) - lastMarginalFitEnd >= ...
            options.marginalRefitEvery
        refitMarginal(index) = true;
        lastMarginalFitEnd = estimationEnd(index);
    end
end

schedule = table(window, estimationStart, estimationEnd, ...
    estimationLength, forecastStart, forecastEnd, forecastLength, ...
    refitMarginal, 'VariableNames', { ...
    'Window', 'EstimationStart', 'EstimationEnd', 'EstimationLength', ...
    'ForecastStart', 'ForecastEnd', 'ForecastLength', 'RefitMarginal'});

end
