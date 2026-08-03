function [results, runDirectory] = runConfiguredExperiment( ...
    data, config, outputRoot)
%RUNCONFIGUREDEXPERIMENT Run and persist one declared experiment.

arguments
    data
    config (1, 1) struct
    outputRoot (1, 1) string = ""
end

results = diss.runExperiment(data, config);
if strlength(outputRoot) == 0
    runDirectory = "";
else
    runDirectory = diss.io.saveRun(results, outputRoot);
end

end
