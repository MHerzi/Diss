function project = createProject()
%CREATEPROJECT Create the MATLAB Project metadata for this repository.

repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
project = matlab.project.createProject(repositoryRoot);
project.Name = "Diss";
addPath(project, fullfile(repositoryRoot, 'src'));
addPath(project, fullfile(repositoryRoot, 'experiments'));
addStartupFile(project, fullfile(repositoryRoot, 'startupProject.m'));

end
