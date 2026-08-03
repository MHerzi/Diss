function results = run(data, config)
%RUN Execute a package-based model pipeline.
%
% Delta-Normal is the first complete model adapter. Other models fail with
% an explicit identifier until their numerical code is integrated.

arguments
    data
    config (1, 1) struct = diss.config.defaults()
end

runTimer = tic;
plan = diss.prepare(data, config);
config = plan.Config;

previousRandomState = rng;
randomStateCleanup = onCleanup(@() rng(previousRandomState));
rng(config.simulation.seed, 'twister');

switch config.execution.mode
    case "full"
        results = runFullSample(plan);
    case "backtest"
        results = runBacktest(plan);
    otherwise
        error('diss:run:InvalidMode', ...
            'Unsupported execution mode: %s.', config.execution.mode);
end

results.Config = config;
results.Schedule = plan.Schedule;
results.DataSummary = dataSummary(plan.Data);
results.Metadata = plan.Metadata;
results.Metadata.ElapsedSeconds = toc(runTimer);

end

function results = runFullSample(plan)
[model, ~] = diss.fitWindow( ...
    plan.Data.Returns, plan.Config, struct());
forecast = diss.forecastWindow(model, 1, plan.Config);

results = struct();
results.Mode = "full";
results.Model = model;
results.Forecast = forecast;
results.Backtest = table();
results.WindowModels = {};
results.Diagnostics = struct();
end

function results = runBacktest(plan)
schedule = plan.Schedule;
config = plan.Config;
dataset = plan.Data;
resultCount = sum(schedule.ForecastLength);

windowColumn = zeros(resultCount, 1);
datasetRow = zeros(resultCount, 1);
observationIndex = zeros(resultCount, 1);
estimationStart = zeros(resultCount, 1);
estimationEnd = zeros(resultCount, 1);
forecastMean = zeros(resultCount, 1);
forecastStdDev = zeros(resultCount, 1);
returnQuantile = zeros(resultCount, 1);
lossVaR = zeros(resultCount, 1);
realizedReturn = zeros(resultCount, 1);
violation = false(resultCount, 1);
fitSeconds = zeros(resultCount, 1);
forecastSeconds = zeros(resultCount, 1);
dependenceExitFlag = nan(resultCount, 1);

if config.output.saveWindowModels
    windowModels = cell(height(schedule), 1);
else
    windowModels = {};
end

state = struct();
outputPosition = 0;
for windowIndex = 1:height(schedule)
    currentWindow = schedule(windowIndex, :);
    trainingRows = currentWindow.EstimationStart:currentWindow.EstimationEnd;
    forecastRows = currentWindow.ForecastStart:currentWindow.ForecastEnd;

    fitTimer = tic;
    context = struct( ...
        'RefitMarginal', currentWindow.RefitMarginal, ...
        'Window', currentWindow.Window);
    [model, state] = diss.fitWindow( ...
        dataset.Returns(trainingRows, :), config, state, context);
    currentFitSeconds = toc(fitTimer);
    forecastTimer = tic;
    forecastReturns = dataset.Returns(forecastRows, :);
    forecast = diss.forecastWindow(model, ...
        currentWindow.ForecastLength, config, forecastReturns);
    currentForecastSeconds = toc(forecastTimer);
    actual = diss.risk.aggregateReturns( ...
        dataset.Returns(forecastRows, :), ...
        config.risk.portfolioWeights, config.risk.returnAggregation);

    positions = (outputPosition + (1:currentWindow.ForecastLength))';
    windowColumn(positions) = currentWindow.Window;
    datasetRow(positions) = forecastRows;
    observationIndex(positions) = dataset.ObservationIndex(forecastRows);
    estimationStart(positions) = currentWindow.EstimationStart;
    estimationEnd(positions) = currentWindow.EstimationEnd;
    forecastMean(positions) = forecast.PortfolioMean;
    forecastStdDev(positions) = forecast.PortfolioStdDev;
    returnQuantile(positions) = forecast.ReturnQuantile;
    lossVaR(positions) = forecast.LossVaR;
    realizedReturn(positions) = actual;
    violation(positions) = actual < forecast.ReturnQuantile;
    fitSeconds(positions) = currentFitSeconds;
    forecastSeconds(positions) = currentForecastSeconds;
    if isfield(model, 'Dependence') && ...
            isfield(model.Dependence, 'ExitFlag')
        dependenceExitFlag(positions) = model.Dependence.ExitFlag;
    end
    outputPosition = outputPosition + currentWindow.ForecastLength;

    if config.output.saveWindowModels
        windowModels{windowIndex} = model;
    end
end

backtest = table(windowColumn, observationIndex, estimationStart, ...
    estimationEnd, forecastMean, forecastStdDev, returnQuantile, ...
    lossVaR, realizedReturn, violation, fitSeconds, ...
    forecastSeconds, dependenceExitFlag, ...
    'VariableNames', { ...
    'Window', 'ObservationIndex', 'EstimationStart', 'EstimationEnd', ...
    'ForecastMean', 'ForecastStdDev', 'ReturnQuantile', 'LossVaR', ...
    'RealizedReturn', 'Violation', 'FitSeconds', 'ForecastSeconds', ...
    'DependenceExitFlag'});

if ~isempty(dataset.Time)
    backtest.ForecastTime = dataset.Time(datasetRow);
    backtest = movevars(backtest, 'ForecastTime', ...
        'After', 'ObservationIndex');
end

results = struct();
results.Mode = "backtest";
results.Model = struct();
results.Forecast = struct();
results.Backtest = backtest;
results.WindowModels = windowModels;
results.Diagnostics = struct( ...
    'ViolationCount', sum(violation), ...
    'ViolationRate', mean(violation), ...
    'ExpectedViolationRate', 1 - config.risk.probability);
end

function summary = dataSummary(dataset)
summary = struct( ...
    'ObservationCount', dataset.ObservationCount, ...
    'SeriesCount', dataset.SeriesCount, ...
    'VariableNames', dataset.VariableNames, ...
    'RemovedRows', dataset.RemovedRows, ...
    'OriginalType', dataset.OriginalType);
end
