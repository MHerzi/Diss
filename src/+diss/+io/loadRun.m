function results = loadRun(runDirectory)
%LOADRUN Load a result artifact created by diss.io.saveRun.

arguments
    runDirectory (1, 1) string
end

resultFile = fullfile(runDirectory, 'results.mat');
if ~isfile(resultFile)
    error('diss:io:MissingResultFile', ...
        'Result file does not exist: %s.', resultFile);
end
contents = load(resultFile, 'results');
results = contents.results;

end
