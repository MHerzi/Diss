function [LogL, StrOutput, CopulaSpec]=fitCopulaVine_grm_1P_forecast(data)
% ----------- Copula Vine fit function ------------------
% this function estimates the parameters of a copula vine with maximum
% likelihood
%
% USAGE:            [LogL, StrOutput]=fitCopulaVine_grm(data)
%
% INPUTS:
% data:             A matrix with U(0,1) margins
% CopulaSpec:       Structure array that contains the various components
%                   for the copula family and optimization procedure. To
%                   obtain run setCopulaVineLLinputs.m
%
% OUTPUTS:
% CopParams:        The estimated Copula Vine parameters
% LogL:             The log likelihood value at the optimum
% StrOutput:        Structure array that contains secondary output
% ----------------------------------------------------------------------
% author: Martin Grziska based on a code of Manthos Vogiatzoglou

% -----------------------------------------------------------------------
% % ---------------------------------------------------------------------

[T,N]=size(data);

% ---------------------------------------------------------------------
% set copula vine inputs
CopulaSpec = setCopulaVineLLinputs_grm(N);
% --------------------------------------------------------------------

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
type=CopulaSpec.type; optimizer=CopulaSpec.optimizer;
% options = optimset('Algorithm','active-set','Display','iter','FunValCheck','on','MaxFunEvals',10000,'TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6);
options = optimset('fmincon');
options = optimset(options,'Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6,'Algorithm','interior-point','Hessian','bfgs','FunValCheck','on','MaxFunEvals',10000);
% ----------------------------------------------------------------------


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
else
    % -------------------------------------------------------------------------
    % constrained optimizations
    % -------------------------------------------------------------------------
    if strcmp(optimizer,'fmincon') == 1
        %         generally: Dependence structure estimated as in Heinen,
        %         Valdesogo, "Asymmetric CAPM dependence for large dimensions: the
        %         Canonical Vine Autoregressive Model"
        if strcmp(type,'t') == 1
            if strcmp(CopulaSpec.corrspec,'static') == 1
                display('the copula parameters is a .5*N*(N-1)x1 vector');
                %             static t-needs starting values for d.o.f and
                %             correlation parameters; unconditional
                %             correlation parameters are used
                %                 correlation estimated with kendall's tau, makes
                %                 comparsion with dynamic models possible
                tau=corr(data,'type','kendall');
                rho=sin(tau*pi/2);
                h=1;
                c=1;
                for i=2:size(rho,2)
                    for j=1:c
                        rho1(h) = rho(i,j);
                        h=h+1;
                    end
                    c=c+1;
                end
                defvals = [10*ones(ParSize,1) rho1'] ;
                lower = [2.1*ones(ParSize,1) -1*ones(ParSize,1)];
                upper = [100*ones(ParSize,1) ones(ParSize,1)];
            elseif strcmp(CopulaSpec.corrspec,'DCC') ==1 || strcmp(CopulaSpec.corrspec,'TVC') == 1
                display('the copula parameters is a .5*N*(N-1)*3 x 1 vector');
                %                 starting values for d.o.f
                defvals=10*ones(ParSize*3,1);
                %                dynamic constraint: a+b<1
                for i=1:3:(ParSize*3)
                    lower(i:i+2) = [0+1e-10 0+1e-10 2.1];
                end
                for i=1:3:(ParSize*3)
                    upper(i:i+2) = [.9999 .9999 100];
                end
            end
        elseif strcmp(type,'Clayton') == 1
            if strcmp(CopulaSpec.corrspec,'static') == 1
                display('the copula parameters is a .5*N*(N-1)x1 vector');
                defvals=ones(ParSize,1);
                lower =10^-4*ones(ParSize,1);
                upper =[];
                A=[];
                b=[];
            elseif strcmp(CopulaSpec.corrspec,'Patton') == 1
                display('the copula parameters is a .5*N*(N-1)x1 vector');
                defvals=.15*ones(ParSize,1);
                lower =[];
                upper =[];
            elseif strcmp(CopulaSpec.corrspec,'DCC') == 1
                display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
                defvals=ones(ParSize*2,1);
                for i=1:2:(ParSize*2)
                    lower(i:i+1) = [0+1e-5 0+1e-5];
                end
                for i=1:2:(ParSize*2)
                    upper(i:i+1) = [1-1e-6 1-1e-6];
                end
            end
        elseif strcmp(type,'SJC') == 1
            display('the copula parameters is a .5*N*(N-1)x2 vector');
            defvals=.15*ones(ParSize,2);
            lower =10^-4*ones(ParSize,2);
            upper =.65*ones(ParSize,2);
        elseif strcmp(type,'Gaussian') == 1
            if (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC')) == 1
                display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
                for i=1:2:(ParSize*2)
                    lower(i:i+1) = [0+1e-10 0+1e-10];
                end
                for i=1:2:(ParSize*2)
                    upper(i:i+1) = [.9999 .9999];
                end
            elseif strcmp(CopulaSpec.corrspec,'static') == 1
                lower = -1;
                upper = 1;
            end
        elseif strcmp(type,'Gumbel') == 1
            if strcmp(CopulaSpec.corrspec,'DCC') == 1
                display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
                defvals = 2*ones(ParSize*2,1);
                for i=1:2:(ParSize*2)
                    lower(i:i+1) = [0+1e-10 0+1e-10];
                end
                for i=1:2:(ParSize*2)
                    upper(i:i+1) = [.9999 .9999];
                end
            elseif strcmp(CopulaSpec.corrspec,'Patton') == 1
                display('the copula parameters is a .5*N*(N-1)x1 vector');
                defvals=.15*ones(ParSize,1);
                lower = [];
                upper =[];
            elseif strcmp(CopulaSpec.corrspec,'static') == 1
                lower = repmat(1.01,ParSize,1);
                upper=[];
            end
        end
        
        % ----------------- the starting value (menu) function ------------------
        
        if strcmp(CopulaSpec.type,'Clayton') == 1
            if strcmp(CopulaSpec.corrspec, 'Patton') == 1;
                %         Form der Startinvals: [omega, alpha; beta]
                startinvals = repmat([.5; .1; .1],ParSize,1);
                A = [];
                b = [];
            elseif strcmp(CopulaSpec.corrspec,'static') == 1
                startinvals = defvals;
            elseif strcmp(CopulaSpec.corrspec,'DCC') == 1;
                %             Es werden die Parameter der Claytion-Copula geschätzt; die
                %             DCC-Parameter werden unter der Gauss-Annahme geschätzt und
                %             dann durch Transformation mittels Kenndals tau in
                %             Clayton-Parameter übertragen; an dieser Stelle müssen die
                %             Clayton-Parameter so restringiert werden, dass die
                %             transformierten Parameter wieder den DCC-Restriktionen
                %             entsprechen
                %             startinvals = repmat([.006; 1.9],ParSize,1);
                startinvals = repmat([.01; .97],ParSize,1);
                A = zeros(ParSize*2,ParSize*2);
                k=1;
                for i = 1:2:ParSize*2
                    A(i,k:k+1)=[1 1];
                    k=k+2;
                end
                b = repmat([1-1e-5; 0],ParSize,1);
            end
        elseif strcmp(CopulaSpec.type,'t') == 1
            if strcmp(CopulaSpec.corrspec, 'DCC')              
                %         Form der Startinvals: [a; b; nu]]
                startinvals = repmat([.01; .9; 10],ParSize,1);
                %         kreiere spezielle Matrix A, so das A*x<b
                A = zeros(ParSize*3,ParSize*3);
                k=1;
                for i = 1:3:ParSize*3
                    A(i,k:k+1)=[1 1];
                    k=k+3;
                end
                b = repmat([1-(1e-10); 0; 0],ParSize,1);
            elseif strcmp(CopulaSpec.corrspec, 'TVC') == 1
                startinvals = repmat([.6; .3; 10],ParSize,1);
                %         kreiere spezielle Matrix A, so das A*x<b
                A = zeros(ParSize*3,ParSize*3);
                k=1;
                for i = 1:3:ParSize*3
                    A(i,k:k+1)=[1 1];
                    k=k+3;
                end
                b = repmat([1-(1e-10); 0; 0],ParSize,1);
            else
                startinvals=defvals;
                A=[];
                b=[];
            end
        elseif strcmp(CopulaSpec.type,'Gaussian') == 1
            if strcmp(CopulaSpec.corrspec, 'DCC')
                %         Form der Startinvals: [a; b]]
                startinvals = repmat([.01; .9],ParSize,1);
                %         kreiere spezielle Matrix A, so das A*x<b
                A = zeros(ParSize*2,ParSize*2);
                k=1;
                for i = 1:2:ParSize*2
                    A(i,k:k+1)=[1 1];
                    k=k+2;
                end
                b = repmat([1-(1e-10); 0],ParSize,1);
            end
            if strcmp(CopulaSpec.corrspec, 'TVC') == 1
                startinvals = repmat([.6; .3],ParSize,1);
                %         kreiere spezielle Matrix A, so das A*x<b
                A = zeros(ParSize*2,ParSize*2);
                k=1;
                for i = 1:2:ParSize*2
                    A(i,k:k+1)=[1 1];
                    k=k+2;
                end
                b = repmat([1-(1e-10); 0; 0],ParSize,1);
            end
        elseif strcmp(CopulaSpec.type,'Gumbel') == 1
            if strcmp(CopulaSpec.corrspec, 'DCC')
                startinvals = repmat([.01; .7],ParSize,1);
                A = zeros(ParSize*2,ParSize*2);
                k=1;
                for i = 1:2:ParSize*2
                    A(i,k:k+1)=[1 1];
                    k=k+2;
                end
                b = repmat([1-(1e-10); 0],ParSize,1);
            elseif strcmp(CopulaSpec.corrspec,'static')
                startinvals = repmat(2,ParSize,1);
                A=[];
                b=[];
            elseif strcmp(CopulaSpec.corrspec,'Patton')
                startinvals = repmat(0.15,ParSize);
            end
        else
            startinvals=inputstartinvals_grm(CopulaSpec,defvals);
            A=[];
            b=[];
        end
        
        % ---------------------------------------------------------------------
        %         CopParams has form:      c12     c23     c34
        %                                  c12|23  c23|34   0
        %                                  c13|24   0       0
        %         for 4 dim (# of params: (1/2) *N(N-1))
        %-----------------------------------------------------------------
        
        
        
        %         ---------------------------------------------------------------
        %                        Estimation procedure
        
        tic
        if strcmp(CopulaSpec.derivatives,'on')==1 && strcmp(CopulaSpec.corrspec,'DCC')==0
            [CopParams, likhood, exitflag, output, lamda, grad, hessian] = fmincon('CopulaVineLL_grm',startinvals,A,b,[],[],lower,upper,[],options,data,CopulaSpec);
            StrOutput.Hessian=hessian;
            StrOutput.Gradient=grad;
            StrOutput.exitflag=exitflag;
            StrOutput.timeinseconds=toc;
        elseif strcmp(CopulaSpec.derivatives,'on')==1 && strcmp(CopulaSpec.corrspec,'DCC')==1
            [CopParams, likhood ,exitflag, output, lamda, grad, hessian] = fmincon('CopulaVineLL_grm',startinvals,A,b,[],[],lower,upper,'DCC_nonlincon_vine',options,data,CopulaSpec);
            StrOutput.Hessian=hessian;
            StrOutput.Gradient=grad;
            StrOutput.exitflag=exitflag;
            StrOutput.timeinseconds=toc;
        elseif strcmp(CopulaSpec.derivatives,'on')==0 && strcmp(CopulaSpec.corrspec,'DCC')==0
            [CopParams, likhood, exitflag]= fmincon('CopulaVineLL_grm',startinvals,A,b,[],[],lower,upper,[],options,data,CopulaSpec);
            StrOutput.exitflag=exitflag;
            StrOutput.timeinseconds=toc;
        elseif strcmp(CopulaSpec.derivatives,'on')==0 && strcmp(CopulaSpec.corrspec,'DCC')==1
            [CopParams, likhood, exitflag]= fmincon('CopulaVineLL_grm',startinvals,A,b,[],[],lower,upper,'DCC_nonlincon_vine',options,data,CopulaSpec);
            StrOutput.exitflag=exitflag;
            StrOutput.timeinseconds=toc;
        end
        if exitflag>0
            display('************************')
            display('succesfull optimization!')
            display('************************')
        else
            display('************************')
            display('unsuccesfull optimization!')
            display('************************')
        end
        LogL = -likhood;
        %         Packt den Spaltenvektor der Parameter in die oben gezeigte Form
        %         der CopParams
        if strcmp(CopulaSpec.corrspec,'Patton') == 1 && strcmp(type,'Clayton') == 1
            [Holder, Rt] = CopulaVineLL_grm_1P_forecast(CopParams,data,CopulaSpec);
            CopParamsOut = cell(ParSize,1);
            k=1;
            for i=1:3:ParSize*3
                CopParamsOut{k} = CopParams(i:i+2,1);
                k=k+1;
            end
            StrOutput.VineParams= crIPmat_grm(CopParamsOut);
            StrOutput.Rt = Rt;
            StrOutput.LogL = LogL;
        elseif strcmp(CopulaSpec.corrspec,'static') && strcmp(type,'Gaussian') == 0
            StrOutput.VineParams=crIPmat(CopParams);
            [Holder, Rt] = CopulaVineLL_grm_1P_forecast(CopParams,data,CopulaSpec);
            StrOutput.Rt = Rt;
            StrOutput.LogL = LogL;
        end
        if strcmp(type,'t')==0 && (strcmp(CopulaSpec.corrspec,'DCC') == 1  || strcmp(CopulaSpec.corrspec,'TVC') == 1)
            [Holder, Rt] = CopulaVineLL_grm_1P_forecast(CopParams,data,CopulaSpec);
            CopParamsOut = cell(ParSize,1);
            k=1;
            for i=1:2:ParSize*2
                CopParamsOut{k} = CopParams(i:i+1,1);
                k=k+1;
            end
            CopParams=crIPmat_grm(CopParamsOut);
            StrOutput.VineParams=CopParams;
            StrOutput.Rt = Rt;
            StrOutput.LogL = LogL;
        elseif strcmp(type,'t')==1 && (strcmp(CopulaSpec.corrspec,'DCC') == 1  || strcmp(CopulaSpec.corrspec,'TVC') == 1)
            [Holder, Rt] = CopulaVineLL_grm_1P_forecast(CopParams,data,CopulaSpec);
            CopParamsOut = cell(ParSize,1);
            k=1;
            for i=1:3:ParSize*3
                CopParamsOut{k} = CopParams(i:i+2,1);
                k=k+1;
            end
            CopParams=crIPmat_grm(CopParamsOut);
            StrOutput.VineParams=CopParams;
            StrOutput.Rt = Rt;
            StrOutput.LogL = LogL;
        end
        StrOutput.exitflag=exitflag;
        StrOutput.timeinseconds=toc;
        StrOutput.timeinmitutes = toc/60;
        StrOutput.LogL = LogL;
        % -------------------------------------------------------------------------
        % unconstrained optimization
        % -------------------------------------------------------------------------
    elseif strcmp(optimizer,'fminunc')
        if strcmp(type,'t')
            display('the copula parameters is a .5*N*(N-1)x1 vector');
            defvals=2.8898*ones(ParSize,1);
        elseif strcmp(type,'Clayton')
            display('the copula parameters is a .5*N*(N-1)x1 vector');
            defvals=-1.9726*ones(ParSize,1);
        elseif strcmp(type,'SJC')
            display('the copula parameters is a .5*N*(N-1)x2 vector');
            defvals=-5.295*ones(ParSize,2);
        end
        % ----------------- the starting value menu function ------------------
        if strcmp(CopulaSpec.corrspec, 'Patton') == 1;
            %         Form der Startinvals: [omega, alpha; beta]
            startinvals = repmat([0; 0; 0],ParSize,1);
        elseif strcmp(CopulaSpec.corrspec, 'DCC') == 1 || strcmp(CopulaSpec.corrspec, 'TVC') == 1
            %         Form der Startinvals: [a; b; nu]]
            startinvals = repmat([.01; .8; 10],ParSize,1);
        else
            startinvals=inputstartinvals(CopulaSpec,defvals);
        end
        % ---------------------------------------------------------------------
        tic
        if strcmp(CopulaSpec.derivatives,'on')==1
            [CopParams, likhood,exitflag,output,grad,hessian]= fminunc('CopulaVineLL_grm',startinvals,options,data,CopulaSpec);
            StrOutput.Hessian=hessian;
            StrOutput.Gradient=grad;
        else
            [CopParams, likhood,exitflag]= fminunc('CopulaVineLL_grm',startinvals,options,data,CopulaSpec);
        end
        if exitflag>0
            display('************************')
            display('succesfull optimization!')
            display('************************')
            LogL=-likhood;
            %         Packt den Spaltenvektor der Parameter in die oben gezeigte Form
            %         der CopParams
            if strcmp(CopulaSpec.corrspec,'Patton') == 1 && strcmp(type,'Clayton') == 1
                [Holder, Rt] = CopulaVineLL_grm(CopParams,data,CopulaSpec);
                CopParamsOut = cell(ParSize,1);
                k=1;
                for i=1:3:ParSize*3
                    CopParamsOut{k} = CopParams(i:i+2,1);
                    k=k+1;
                end
                StrOutput.VineParams= crIPmat_grm(CopParamsOut);
                StrOutput.CondRt = Rt;
            elseif strcmp(CopulaSpec.corrspec,'static')
                StrOutput.VineParams=crIPmat(CopParams);
            end
            
            if strcmp(type,'Gaussian')==1 && (strcmp(CopulaSpec.corrspec,'DCC') == 0  || strcmp(CopulaSpec.corrspec,'TVC') == 0)
                CopParams=crIPmat(.0001+.85./(1+exp(-CopParams)));
            elseif strcmp(type,'t')==1 && strcmp(CopulaSpec.corrspec,'static') == 1
                StrOutput.VineParams=crIPmat(2.1+exp(CopParams));
            end
            
            if strcmp(type,'Gaussian')==1 && (strcmp(CopulaSpec.corrspec,'DCC') == 1  || strcmp(CopulaSpec.corrspec,'TVC') == 1)
                [Holder, Rt] = CopulaVineLL_grm(CopParams,data,CopulaSpec);
                %             Wandle die Copula-Parameter in eine obere Dreiecksmatrix um
                CopParamsOut = cell(ParSize,1);
                k=1;
                for i=1:2:ParSize*2
                    CopParamsOut{k} = CopParams(i:i+1,1);
                    k=k+1;
                end
                CopParams=crIPmat_grm(CopParamsOut);
                StrOutput.VineParams=CopParams;
                StrOutput.CondRt = Rt;
            elseif strcmp(type,'t')==1 && (strcmp(CopulaSpec.corrspec,'DCC') == 1  || strcmp(CopulaSpec.corrspec,'TVC') == 1)
                [Holder, Rt] = CopulaVineLL_grm(CopParams,data,CopulaSpec);
                %             Wandle die Copula-Parameter in eine
                %             obere Dreiecksmatrix um
                CopParamsOut = cell(ParSize,1);
                k=1;
                for i=1:3:ParSize*3
                    CopParamsOut{k} = CopParams(i:i+2,1);
                    k=k+1;
                end
                CopParams=crIPmat_grm(CopParamsOut);
                StrOutput.VineParams=CopParams;
                StrOutput.CondRt = Rt;
                if strcmp(type,'t')==0 && (strcmp(CopulaSpec.corrspec,'DCC') == 0  && strcmp(CopulaSpec.corrspec,'TVC') == 0)
                    StrOutput.VineParams=crIPmat(.0001+.85./(1+exp(-CopParams)));
                elseif strcmp(type,'t')==1 && strcmp(CopulaSpec.corrspec,'static') == 1
                    StrOutput.VineParams=crIPmat(2.1+exp(CopParams));
                    StrOutput.LogL = LogL;
                end
            end
            
            StrOutput.exitflag=exitflag;
            StrOutput.timeinseconds=toc;
            StrOutput.timeinmitutes = toc/60;
        else
            error('unsuccesfull optimization, use fmincon or change initial values')
        end
    end
end
