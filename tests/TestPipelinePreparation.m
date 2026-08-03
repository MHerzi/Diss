function tests = TestPipelinePreparation
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repositoryRoot);
end

function testFullSamplePlanUsesValidatedNumericData(testCase)
    data = reshape(1:30, 10, 3);

    plan = diss.prepare(data, diss.config.defaults());

    verifyEqual(testCase, plan.Data.Returns, double(data));
    verifyEqual(testCase, plan.Data.ObservationCount, 10);
    verifyEqual(testCase, plan.Data.SeriesCount, 3);
    verifyEmpty(testCase, plan.Schedule);
    verifyEqual(testCase, plan.Config.execution.mode, "full");
end

function testExpandingScheduleHasSharedWindowBoundaries(testCase)
    config = backtestConfig();
    config.backtest.marginalRefitEvery = 6;

    plan = diss.prepare(randn(20, 2), config);

    verifyEqual(testCase, plan.Schedule.EstimationStart, [1; 1; 1]);
    verifyEqual(testCase, plan.Schedule.EstimationEnd, [10; 13; 16]);
    verifyEqual(testCase, plan.Schedule.ForecastStart, [11; 14; 17]);
    verifyEqual(testCase, plan.Schedule.ForecastEnd, [13; 16; 19]);
    verifyEqual(testCase, plan.Schedule.RefitMarginal, ...
        [true; false; true]);
end

function testRollingScheduleUsesOnlyConfiguredHistory(testCase)
    config = backtestConfig();
    config.backtest.windowType = "rolling";
    config.backtest.rollingWindow = 8;

    plan = diss.prepare(randn(20, 2), config);

    verifyEqual(testCase, plan.Schedule.EstimationStart, [3; 6; 9]);
    verifyEqual(testCase, plan.Schedule.EstimationLength, [8; 8; 8]);
end

function testPartialFinalWindowIsExplicit(testCase)
    config = backtestConfig();
    config.backtest.includePartialFinalWindow = true;

    plan = diss.prepare(randn(20, 2), config);

    verifyEqual(testCase, plan.Schedule.ForecastStart, [11; 14; 17; 20]);
    verifyEqual(testCase, plan.Schedule.ForecastEnd, [13; 16; 19; 20]);
    verifyEqual(testCase, plan.Schedule.ForecastLength, [3; 3; 3; 1]);
end

function testMissingRowsRequireExplicitPolicy(testCase)
    data = randn(12, 2);
    data(4, 1) = NaN;

    verifyError(testCase, @() diss.prepare(data, diss.config.defaults()), ...
        'diss:data:NonfiniteReturns');

    config = diss.config.defaults();
    config.data.missingAction = "omitRows";
    plan = diss.prepare(data, config);

    verifyEqual(testCase, plan.Data.RemovedRows, 4);
    verifyEqual(testCase, plan.Data.ObservationCount, 11);
    verifyFalse(testCase, any(~isfinite(plan.Data.Returns), 'all'));
end

function testLegacySpecMapsToNestedConfiguration(testCase)
    spec = struct( ...
        'purpose', 'backtest', ...
        'ModelType', 'MultiCopula', ...
        'DynamicType', 'ADCC', ...
        'ForecastStart', 10, ...
        'ForecastNumb', 3, ...
        'uniforecastP', 6, ...
        'SimNumb', 5000, ...
        'beta', 0.95, ...
        'univariate', 'on', ...
        'uniBacktest', 'on', ...
        'arlag', 2, ...
        'const', 1, ...
        'archP', 1, ...
        'garchQ', 1);

    config = diss.config.fromLegacySpec(spec);
    plan = diss.prepare(randn(20, 2), config);

    verifyEqual(testCase, plan.Config.model.kind, "multiCopula");
    verifyEqual(testCase, plan.Config.model.dynamic, "ADCC");
    verifyEqual(testCase, plan.Config.backtest.initialWindow, 10);
    verifyEqual(testCase, plan.Config.backtest.forecastHorizon, 3);
    verifyEqual(testCase, plan.Config.simulation.numPaths, 5000);
    verifyEqual(testCase, plan.Config.risk.probability, 0.95);
end

function testBacktestRequiresExplicitInitialWindow(testCase)
    config = diss.config.defaults();
    config.execution.mode = "backtest";

    verifyError(testCase, @() diss.prepare(randn(20, 2), config), ...
        'diss:config:MissingInitialWindow');
end

function config = backtestConfig()
    config = diss.config.defaults();
    config.execution.mode = "backtest";
    config.backtest.initialWindow = 10;
    config.backtest.forecastHorizon = 3;
    config.backtest.stepSize = 3;
end
