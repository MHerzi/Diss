function [LogL, StrOutput, CopulaSpec, phi] = fitCopulaVine_multistep_mix_grm(data,CopulaSpec,Cop_Stat,PrtFig);
% ----------- Copula Vine fit function ------------------
% this function estimates the parameters of a copula vine with maximum
% likelihood; estimation procedure: multi-step
%
% USAGE:            [LogL, StrOutput, CopulaSpec] = fitCopulaVine_grm(data,CopulaSpec)
%
% INPUTS:
% data:             A matrix with U(0,1) margins
% CopulaSpec:       Structure array that contains the various components
%                   for the copula family and optimization procedure. To
%                   obtain run setCopulaVineLLinputs.m
% PrtFig:           if 'on' figures will be printed
%
% OUTPUTS:
% CopParams:        The estimated Copula Vine parameters
% LogL:             The log likelihood value at the optimum
% StrOutput:        Structure array that contains secondary output
% ----------------------------------------------------------------------
% author: Martin Grziska
% Date: 09/06/2010

% -----------------------------------------------------------------------
% ---------------------------------------------------------------------
% tic;
[T,N]=size(data);

% ---------------------------------------------------------------------
% set copula vine inputs
if  nargin<2 || isempty(CopulaSpec)
    CopulaSpec = setCopulaVineLLinputs_grm(N);
end
% --------------------------------------------------------------------


if strcmp(CopulaSpec.decomposition,'DVine-Mix') == 0
    % rotatded Clayton: transform data; everything else is same as Clayton:
    % seit Copula-type to Clayton back
    if strcmp(CopulaSpec.type,'Rotated Clayton')
        data=1-data;
        CopulaSpec.type = 'Clayton';
        CoplaSpec.rotClayton = 1;
    end
    % ----------------------------------------------------------------------
    ParSize = .5*(N-1)*N;
    % some error checking
    if N<3
        error('error in t x k datamatrix: k < 3; vines make only sense if k >= 3!')
    end
    if isempty(CopulaSpec)==1 || isstruct(CopulaSpec)==0
        error('run the setCopulaVineLLinputs.m funtion to obtain the CopulaSpec input');
    end
    if min(min(data))<0 || max(max(data))>1
        display('your data is not uniform. It is transformed to uniform with the empiricalCDF')
        data=empiricalCDF(data);
    end
    if strcmp(CopulaSpec.use,'EstimateStartinVals')==1
        display(' ')
        display('this function fits the vine at one step')
        display('to obtain starting values run EstInvals4CopulaVine.m')
        display('press crtl + C to abort current function')
        pause(5)
        display(' ')
    end
    % -----------------------------------------------------------------------
    
    % ----------------------------------------------------------------------
    type=CopulaSpec.type;
    % ----------------------------------------------------------------------
    
    % ------------------------------------------------------------------------
    
    % ------------------------------------------------------------------------
    % Estimation
    % -----------------------------------------------------------------------
    if strcmp(type,'Gaussian') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
        %     static Gaussian Copula doesn't need any optimization, unconditional
        %     correlation matrix is the same, needs just LogL
        [LogL,Rt]=CopulaVineLLGaussStatic(data,CopulaSpec);
        Rt1=cell(N-1,N-1);
        c=N-1;
        for i=1:N-1
            for j=1:c
                Rt1{i,j} = Rt{i,j}(1);
            end
            c=c-1;
        end
        clear Rt
        Rt = Rt1;
        StrOutput.Rt = Rt;
        StrOutput.LogL = LogL;
        StrOutput.VineParams = Rt;
        return
    end
    % -------------------------------------------------------------------------
    % optimizations
    % -------------------------------------------------------------------------
    %         generally: Dependence structure estimated as in Heinen,
    %         Valdesogo, "Asymmetric CAPM dependence for large dimensions: the
    %         Canonical Vine Autoregressive Model"
    
    
    
    % -------------------------------------------------------------------------
    %                        Estimation procedure
    
    [StrOutput,phi] = Est_Vine_multistep(CopulaSpec,N,data,ParSize,T);
    
    
    %         -----------------------------------------------------------------
    %     put estimated parameters in column vector which can be used for
    %     CopulaVineLL_grm.m
    c=N-1;
    g=1;
    CopParams = zeros(size(phi{1,1},1)*ParSize,1);
    for i=1:N-1
        for j=1:c
            CopParams(g:g+size(phi{1,1},1)-1) = phi{i,j};
            g=g+size(phi{1,1},1);
        end
        c=c-1;
    end
    %         -----------------------------------------------------------------
    %         berechne Tstat und stderrors
    if strcmp(CopulaSpec.stderrors,'on') == 1
        %             if hessian is not computed by fmincon TstatVine calculates
        %             own hessian
        if strcmp(CopulaSpec.derivatives,'off')
            hessian = [];
        end
        [Tstat,stderrors] = TstatVine(CopParams,data,CopulaSpec,hessian);
        StrOutput.Tstat=Tstat;
        StrOutput.stderrors=stderrors;
    end
    %     ---------------------------------------------------------------------
    % evaluate LogL, Rt, and (complete-not sum) likelihoods
    if strcmp(CopulaSpec.corrspec,'Patton') == 1 && (strcmp(type,'Clayton') ||  strcmp(type,'Gumbel'))
        [LogL, Rt, likelihoods] = CopulaVineLL_grm(CopParams,data,CopulaSpec);
        %     make transformations
        if strcmp(CopulaSpec.optimizer,'fminunc') && strcmp(CopulaSpec.type,'Clayton')
            CopParams=.001+10./(1+exp(-CopParams));
        elseif strcmp(CopulaSpec.optimizer,'fminunc') && strcmp(CopulaSpec.type,'Gumbel')
            CopParams=1.001+10./(1+exp(-CopParams));
        end
        CopParamsOut = cell(ParSize,1);
        k=1;
        for i=1:3:ParSize*3
            CopParamsOut{k} = CopParams(i:i+2,1);
            k=k+1;
        end
        StrOutput.VineParams= crIPmat_grm(CopParamsOut);
        StrOutput.Rt = Rt;
        StrOutput.LogL = LogL;
        StrOutput.likelihoods = likelihoods;
    elseif strcmp(CopulaSpec.corrspec,'static') && strcmp(type,'Gaussian') == 0
        StrOutput.VineParams=crIPmat(CopParams);
        [LogL, Rt, likelihoods] = CopulaVineLL_grm(CopParams,data,CopulaSpec);
        StrOutput.Rt = Rt;
        StrOutput.LogL = LogL;
        StrOutput.likelihoods = likelihoods;
    end
    if strcmp(type,'t')==0 && (strcmp(CopulaSpec.corrspec,'DCC') == 1  || strcmp(CopulaSpec.corrspec,'TVC') == 1)
        [LogL, Rt, likelihoods] = CopulaVineLL_grm(CopParams,data,CopulaSpec);
        CopParamsOut = cell(ParSize,1);
        k=1;
        for i=1:2:ParSize*2
            CopParamsOut{k} = CopParams(i:i+1,1);
            k=k+1;
        end
        CopParamsOut=crIPmat_grm(CopParamsOut);
        StrOutput.VineParams=CopParamsOut;
        StrOutput.Rt = Rt;
        StrOutput.LogL = LogL;
        StrOutput.likelihoods = likelihoods;
    elseif strcmp(type,'t') == 1 && (strcmp(CopulaSpec.corrspec,'DCC') == 1  || strcmp(CopulaSpec.corrspec,'TVC') == 1)
        [LogL, Rt, likelihoods] = CopulaVineLL_grm(CopParams,data,CopulaSpec);
        CopParamsOut = cell(ParSize,1);
        k=1;
        for i=1:3:ParSize*3
            CopParamsOut{k} = CopParams(i:i+2,1);
            k=k+1;
        end
        CopParamsOut=crIPmat_grm(CopParamsOut);
        StrOutput.VineParams=CopParamsOut;
        StrOutput.Rt = Rt;
        StrOutput.LogL = LogL;
        StrOutput.likelihoods = likelihoods;
    end
else
    [StrOutput,phi] = Est_Vine_multistep_mix(CopulaSpec,N,data,T,Cop_Stat,PrtFig);
    LogL=[];
end
