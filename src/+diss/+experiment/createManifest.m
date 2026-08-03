function manifest = createManifest(plan, metadata)
%CREATEMANIFEST Record the inputs needed to reproduce an experiment.

arguments
    plan (1, 1) struct
    metadata (1, 1) struct
end

configHash = diss.io.hashValue(plan.Config);
returnsHash = diss.io.hashArray(plan.Data.Returns);
if isempty(plan.Data.Time)
    timeValues = strings(0, 1);
else
    timeValues = string(plan.Data.Time);
end
dataHash = diss.io.hashValue(struct( ...
    'ReturnsHash', returnsHash, ...
    'ObservationIndex', plan.Data.ObservationIndex, ...
    'Time', timeValues, ...
    'VariableNames', plan.Data.VariableNames));
manifest = struct();
manifest.SchemaVersion = plan.Config.schemaVersion;
experimentHash = diss.io.hashValue(struct( ...
    'ConfigHash', configHash, 'DataHash', dataHash));
manifest.ExperimentId = extractBefore(experimentHash, 17);
manifest.GitCommit = currentGitCommit();
manifest.MatlabRelease = metadata.MatlabRelease;
created = metadata.Created;
created.Format = 'yyyy-MM-dd''T''HH:mm:ss.SSS';
manifest.Created = string(created);
manifest.RandomSeed = metadata.RandomSeed;
manifest.ConfigHash = configHash;
manifest.DataHash = dataHash;
manifest.ObservationCount = plan.Data.ObservationCount;
manifest.SeriesCount = plan.Data.SeriesCount;
manifest.Products = productVersions();

end

function commit = currentGitCommit()
[status, output] = system('git rev-parse HEAD');
if status == 0
    commit = strtrim(string(output));
else
    commit = "unknown";
end
end

function products = productVersions()
installed = ver;
products = struct('Name', {installed.Name}, ...
    'Version', {installed.Version}, ...
    'Release', {installed.Release});
end
