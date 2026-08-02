function [CopParams, LogL, StrOutput]=fitCopula_grm(data)
% ----------- Copula  fit function ------------------
% this function estimates the parameters of a copula with maximum
% likelihood
% INPUTS:
% data:             A matrix with U(0,1) margins
% CopulaSpec:       Structure array that contains the various components
%                   for the copula family and optimization procedure. To
%                   obtain run setCopulaLLinputs.m
% OUTPUTS:
% CopParams:        The estimated Copula Vine parameters
% LogL:             The log likelihood value at the optimum
% StrOutput:        Structure array that contains secondary output
% ----------------------------------------------------------------------
% author: Martin Grziska based on a code of Manthos Vogiatzoglou
% -----------------------------------------------------------------------
[T,N]=size(data);

CopulaSpec = setCopulaLLinputs_grm(N);

% needed becauase of mix of code
CopulaSpec.corrspec=CopulaSpec.depspec;

% some error checking
if isempty(CopulaSpec)==1 || isstruct(CopulaSpec)==0
    error('run the setCopulaLLinputs.m funtion to obtain the CopulaSpec input');
end
if min(min(data))<0 || max(max(data))>1
    display('your data is not uniform. It is transformed to uniform with the empiricalCDF')
    data=empiricalCDF(data);
end
% ----------------------------------------------------------------------
type=CopulaSpec.type; corrspec=CopulaSpec.depspec; optimizer=CopulaSpec.optimizer;
options = optimset('Algorithm','active-set','Display','iter','MaxFunEvals',10000,'TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6);
tic
% -----------------------------------------------------------------------
% constrained optimizations
% ------------------------------------------------------------------------
if strcmp(optimizer,'fmincon')==1
    if strcmp(type,'t')==1 && (strcmp(corrspec,'DCC')==1 || strcmp(corrspec,'TVC')==1)
        display('the parameters is a 3x1 vector with the d.o.f. parameter put first')
        defvals=[22.0855;.0174;.9783];
        lower =[2.01;.0001*ones(2,1)]; 
        upper =[200; ones(2,1)];
        A=[0 ones(1,2)];
        b=0.9999;
    elseif strcmp(type,'t')==1 && strcmp(corrspec,'static')==1
        display('the parameter is scalar, the degree of freedom.')
        defvals=22.0855;
        lower =2.01; 
        upper =200; 
        A=[];
        b=[];
    end
    if strcmp(type,'Clayton')==1 && strcmp(corrspec,'Patton')==1
        display('the parameters vector is a 3x1 vector.');
        defvals=[-1;-1;.5];
        lower =-18*ones(3,1); 
        upper =18*ones(3,1);
        A=[];
        b=[];
    elseif strcmp(type,'Clayton')==1 && strcmp(corrspec,'static')==1
        display('the parameter is a scalar in (0,1)');
        val=corr(data,'type','Kendall'); defvals=val(1,2);
        lower = 10^-5;
        upper=.8;
        A=[];
        b=[];
    end
    if strcmp(type,'SJC')==1 && strcmp(corrspec,'Patton')==1
        display('the parameters vector is a 3x2 vector, first column for upper tail');
        defvals=[-2 -2; -1 -1; .5 .5];
        A=[]; b=[];
        lower=-18*ones(3,2); upper=18*ones(3,2);
    end
    if strcmp(type,'SJC')==1 && strcmp(corrspec,'static')==1
        display('the parameters is a 2x1 vector, first input for the upper tail. Both in (0,1).');
        defvals=[.3;.3];
        A=[]; b=[];
        lower=.0001*ones(2,1); upper=.65*ones(2,1);
    end   
    if strcmp(type,'Gaussian')==1 && (strcmp(corrspec,'DCC')==1 || strcmp(corrspec,'TVC')==1)
        display('the parameters vector is a 2x1 vector');
        defvals=[.0174;.9783];
        lower =.0001*ones(2,1); 
        upper =ones(2,1);
        A=ones(1,2);
        b=0.9999;
    end
    % ----------------- the starting value menu function ----------------------
    startinvals=inputstartinvals(CopulaSpec,defvals);
    % ---------------------------------------------------------------------
    % ----------------- the optimization function -------------------------
    if strcmp(CopulaSpec.derivatives,'on')==1
    [CopParams, likhood,exitflag,output,lamda,grad,hessian]= fmincon('CopulaLL',startinvals,A,b,[],[],lower,upper,[],options,data,CopulaSpec);
    StrOutput.Hessian=hessian;
    StrOutput.Gradient=grad;
    else
    [CopParams, likhood,exitflag]= fmincon('CopulaLL',startinvals,A,b,[],[],lower,upper,[],options,data,CopulaSpec);
    end
    % ----------------- commands to create output -------------------------
    StrOutput.exitflag=exitflag;
    StrOutput.timeinseconds=toc;
    if exitflag>0
        display('succesfull optimization!')
            LogL=-likhood;
    end
end
% ------------------------------------------------------------------------
% unconstrained optimization
% ------------------------------------------------------------------------
if strcmp(optimizer,'fminunc')==1
    if strcmp(type,'t')==1 && (strcmp(corrspec,'DCC')==1 || strcmp(corrspec,'TVC')==1)
        display('the parameters vector is 3x1, the degree of freedom is put first');
        defvals=[3;2;15];
    elseif strcmp(type,'t')==1 && strcmp(corrspec,'static')==1
        display('the parameter is scalar, the degree of freedom.')
        defvals=3;
    end
    if strcmp(type,'Clayton')==1 && strcmp(corrspec,'Patton')==1
        display('the parameters vector is a 3x1 vector.');
        defvals=[-1;-1;.5];
    elseif strcmp(type,'Clayton')==1 && strcmp(corrspec,'static')==1
        display('the copula parameter is a scalar in (0,1)');
        val=corr(data,'type','Kendall'); defvals=tan(pi*(val(1,2)-.5));
    end
    if strcmp(type,'Gaussian')==1 && (strcmp(corrspec,'DCC')==1 || strcmp(corrspec,'TVC')==1)
        display('the parameters vector is a 2x1 vector.');
        defvals=[2;15];
    end
    if strcmp(type,'SJC')==1 && strcmp(corrspec,'Patton')==1
        display('the parameters vector is a 3x2 vector, first column for upper tail');
        defvals=[-2 -2; -1 -1; .5 .5];
    elseif strcmp(type,'SJC')==1 && strcmp(corrspec,'static')==1
        display('the parameters is a 2x1 vector, first input for the upper tail. Both in (0,1).');
        defvals=[-1.2;-1.2];
    end
    % ----------------- the starting value menu ---------------------------
    startinvals=inputstartinvals(CopulaSpec,defvals);
    % ---------------------------------------------------------------------
    % ----------------- the optimization function -------------------------
    if strcmp(CopulaSpec.derivatives,'on')==1
    [CopParams, likhood,exitflag,output,grad,hessian]= fminunc('CopulaLL',startinvals,options,data,CopulaSpec);
    StrOutput.Hessian=hessian;
    StrOutput.Gradient=grad;
    else
    [CopParams, likhood,exitflag]= fminunc('CopulaLL',startinvals,options,data,CopulaSpec);
    end
    % ----------------- commands to create output -------------------------
    StrOutput.exitflag=exitflag;
    StrOutput.timeinseconds=toc;
    if exitflag>0
        LogL=-likhood;
        display('succesfull optimization!')
        if strcmp(corrspec,'TVC')==1 || strcmp(corrspec,'DCC')==1
            if size(CopParams,1)==3
                ARCHparam=CopParams(2)^2/(1+CopParams(2)^2+CopParams(3)^2);
                GARCHparam=CopParams(3)^2/(1+CopParams(2)^2+CopParams(3)^2);
                CopParams(1)=2.01+exp(CopParams(1));
                StrOutput.ConstrParams=[CopParams(1);ARCHparam;GARCHparam];
            elseif size(CopParams,1)==2
                ARCHparam=CopParams(1)^2/(1+CopParams(2)^2+CopParams(1)^2);
                GARCHparam=CopParams(2)^2/(1+CopParams(2)^2+CopParams(1)^2);
                StrOutput.ConstrParams=[ARCHparam;GARCHparam];
            end
        elseif strcmp(corrspec,'static')==1 && strcmp(type,'Clayton')==1
            ConstrParams=.0001+.85./(1+exp(-CopParams));
            StrOutput.ConstrParams=ConstrParams;
        elseif strcmp(corrspec,'static')==1 && strcmp(type,'SJC')==1
            ConstrParams=.0001+.85./(1+exp(-CopParams));
            StrOutput.ConstrParams=ConstrParams;
        elseif strcmp(corrspec,'static')==1 && strcmp(type,'t')==1   
            ConstrParams=2.01+exp(CopParams);
            StrOutput.ConstrParams=ConstrParams;
        end
    end
end
% ------------------- semidefinite programming optimization ---------------
if strcmp(optimizer,'SDPT3')==1
    if max(max(data))<1 && min(min(data))>0
        data=norminv(data);
    end
    if strcmp(corrspec,'static')==1 
    W=(T-1)*cov(data)/T;
    C = sdpvar(N,N); 
    LogL=.5*T*(-logdet(C)+trace((C-eye(N))*W)); 
    %func=.5*T*(-logdet(C)+trace((C-eye(n))*W)); %+n*log(2*pi)
    F=C>=0;
    sol=solvesdp(F,LogL,sdpsettings('solver','sdpt3'));
    conc=double(C); CopParams=conc;
    [var, R]=cov2corr(inv(conc));
    StrOutput.correlation=R;
    elseif strcmp(corrspec,'Penalized')==1 
    W=(T-1)*cov(data)/T;
    D = sdpvar(N,N,'symmetric');
    Chat=inv(corr(data));
    t=input('input the shrinkage parameter, t. For a naive default guess press enter:');
    if isempty(t)==1
        t=.75*sum(vecl(corr(data)));
    end
    LogL=.5*T*(-logdet(D.*Chat)+trace((D.*Chat-eye(N))*W)); 
    F=t>=sum(2*(vecl(D)));
    F=[F,vecl(D)>=0];
    sol=solvesdp(F,LogL,sdpsettings('solver','sdpt3'));
    StrOutput.ShrinkageMat=double(D);
    conc=double(D.*Chat);
    CopParams=conc;
    [var, R]=cov2corr(inv(conc));
    StrOutput.correlation=R;
    StrOutput.ShrnkageParameter=t;
    end
LogL=-double(LogL);
pc=pcfromcons(conc); StrOutput.PartialCor=pc;
x=vecl(R); 
for i=1:N*(N-1)/2
   if abs(x(i))<.0001
       x(i)=0;
   end
end
y=ivecl(x,'full');
StrOutput.ShrCor=y;
StrOutput.yalmip=sol;
end           