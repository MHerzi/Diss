function plan = prepare(data, config)
%PREPARE Validate inputs and build a reproducible pipeline execution plan.
%
% plan = diss.prepare(data, config) is the first migration stage from the
% legacy MainFile script. It performs no estimation and therefore cannot
% change numerical model results.

arguments
    data
    config (1, 1) struct = diss.config.defaults()
end

config = diss.config.validate(config, data);
dataset = diss.data.prepare(data, config.data);

% Omitting incomplete rows can change the legal backtest range. Validate a
% second time against the actual dataset used by the pipeline.
config = diss.config.validate(config, dataset.Returns);

if config.execution.mode == "backtest"
    schedule = diss.backtest.createSchedule( ...
        dataset.ObservationCount, config.backtest);
else
    schedule = table();
end

plan = struct();
plan.Config = config;
plan.Data = dataset;
plan.Schedule = schedule;
plan.Metadata = struct( ...
    'SchemaVersion', config.schemaVersion, ...
    'MatlabRelease', string(version('-release')), ...
    'Created', datetime('now'), ...
    'RandomSeed', config.simulation.seed);

end
