function startupProject()
%STARTUPPROJECT Configure the controlled dissertation development path.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'));
addpath(fullfile(projectRoot, 'experiments'));
cd(projectRoot);

end
