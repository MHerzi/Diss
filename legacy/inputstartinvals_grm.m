function startinvals=inputstartinvals_grm(CopulaSpec,defvals)
if strcmp(CopulaSpec.ll,'copula')==1 && isfield(CopulaSpec,'decomposition')==0
    aa=menu('define starting values','input defaults','keyboard input','already in workspace');
    if aa==1
        startinvals=defvals;
    elseif aa==2
        startinvals=input('type the starting values for the optimization:');
    elseif aa==3
        theta00=input('write the name of the variable in the workspace that contains the starting values:','s');
        startinvals=evalin('base',theta00);
    end
elseif strcmp(CopulaSpec.ll,'copulavine')==1 && strcmp(CopulaSpec.use,'EstimateStartinVals')==1
    startinvals=defvals;
elseif strcmp(CopulaSpec.ll,'copulavine')==1
    bb=menu('define starting values','input defaults','already in workspace');
    if bb==1
        startinvals=defvals;
    elseif bb==2
        theta00=input('write the name of the variable in the workspace that contains the starting values:','s');
        startinvals=evalin('base',theta00);
    end
elseif strcmp(CopulaSpec.ll,'FVine')==1
    startinvals=defvals;
end