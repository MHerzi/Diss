function results = runAllTests()
%RUNALLTESTS Execute every unit, regression and integration test.

repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repositoryRoot);
addpath(fullfile(repositoryRoot, 'src'));
suite = testsuite(fullfile(repositoryRoot, 'tests'), ...
    'IncludeSubfolders', true);
results = run(suite);
assertSuccess(results);

end
