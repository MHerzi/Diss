function runDirectory = saveRun(results, outputRoot)
%SAVERUN Save results and a human-readable reproducibility manifest.

arguments
    results (1, 1) struct
    outputRoot (1, 1) string = "results"
end

if ~isfield(results, 'Manifest') || ...
        ~isfield(results.Manifest, 'ExperimentId')
    error('diss:io:MissingManifest', ...
        'Only results produced by diss.runExperiment can be saved.');
end

timestamp = string(datetime('now', ...
    'Format', 'yyyyMMdd''T''HHmmss_SSS'));
runName = timestamp + "_" + results.Manifest.ExperimentId;
runDirectory = string(fullfile(outputRoot, runName));
if isfolder(runDirectory) || isfile(runDirectory)
    error('diss:io:RunAlreadyExists', ...
        'Run output already exists: %s.', runDirectory);
end
mkdir(runDirectory);

resultFile = fullfile(runDirectory, 'results.mat');
save(resultFile, 'results', '-v7.3');
manifestFile = fullfile(runDirectory, 'manifest.json');
writeText(manifestFile, jsonencode(results.Manifest));

end

function writeText(fileName, contents)
[fileId, message] = fopen(fileName, 'w', 'n', 'UTF-8');
if fileId < 0
    error('diss:io:CannotWriteManifest', ...
        'Cannot open %s: %s.', fileName, message);
end
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s\n', contents);
end
