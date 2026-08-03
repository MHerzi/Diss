function cleanup = setupLegacyPath()
%SETUPLEGACYPATH Add the historical tree explicitly for reproduction work.

legacyRoot = fileparts(mfilename('fullpath'));
previousPath = path;
addpath(genpath(legacyRoot));
cleanup = onCleanup(@() path(previousPath));

end
