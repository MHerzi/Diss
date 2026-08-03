function [results, finalState] = runExperiment(data, config, initialState)
%RUNEXPERIMENT Execute the canonical dissertation experiment pipeline.

arguments
    data
    config (1, 1) struct = diss.config.defaults()
    initialState (1, 1) struct = struct()
end

[results, finalState] = diss.experiment.run(data, config, initialState);

end
