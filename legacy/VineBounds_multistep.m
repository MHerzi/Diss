function [lower, upper, defvals] = VineBounds_multistep(data,CopulaSpec,ParSize)

% Helper function for fitCopulaVine_grm.m. Claculates upper bounds, lower
% bounds, and default starting values.
%
% USAGE:
%        [lower, upper, defvals] = VineBounds(data,CopulaSpec,ParSize)
%
% INPUTS:
%        data:       t x k array fof unif(0,1) variables
%  CopulaSpec:       structure from setCopulaVineLLinputs.m
%     ParSize:       # of (conditional) bivariate copulas to be estimated
%                    in vine

type = CopulaSpec.type;
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
        lower = [2.1*ones(ParSize,1) -1*ones(ParSize,1)+1e-6];
        upper = [150*ones(ParSize,1) ones(ParSize,1)-1e-6];
    elseif strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1
        display('the copula parameters is a .5*N*(N-1)*3 x 1 vector');
        %                 starting values for d.o.f
        defvals=10*ones(ParSize*3,1);
        %                dynamic constraint: a+b<1
        for i=1:3:(ParSize*3)
            lower(i:i+2) = [0+1e-6 0+1e-6 2.1];
        end
        for i=1:3:(ParSize*3)
            upper(i:i+2) = [.99999 .99999 150];
        end
    elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
        display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
        for i=1:3:(ParSize*4)
            lower(i:i+3) = [0+1e-6 0+1e-6 0+1e-6 2.1];
        end
        for i=1:3:(ParSize*4)
            upper(i:i+2) = [.9999 .9 .9999 150];
        end
    end
elseif strcmp(type,'Clayton') == 1
    if strcmp(CopulaSpec.corrspec,'static') == 1
        display('the copula parameters is a .5*N*(N-1)x1 vector');
        defvals=ones(ParSize,1);
        lower = zeros(ParSize,1)+1e-6;
        upper =50;
    elseif strcmp(CopulaSpec.corrspec,'Patton') == 1
        display('the copula parameters is a .5*N*(N-1)x1 vector');
        defvals=.15*ones(ParSize,1);
        lower =[];
        upper =[];
    elseif strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 
        display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
        defvals=ones(ParSize*2,1);
        for i=1:2:(ParSize*2)
            lower(i:i+1) = [0+1e-5 0+1e-5];
        end
        for i=1:2:(ParSize*2)
            upper(i:i+1) = [1-1e-6 1-1e-6];
        end
    elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
        display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
        defvals=ones(ParSize*3,1);
        for i=1:3:(ParSize*3)
            lower(i:i+2) = [0+1e-6 0+1e-6 0+1e-6];
        end
        for i=1:3:(ParSize*3)
            upper(i:i+2) = [.9999 .9 .9999];
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
    elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
        display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
        for i=1:3:(ParSize*3)
            lower(i:i+2) = [0+1e-6 0+1e-6 0+1e-6];
        end
        for i=1:3:(ParSize*3)
            upper(i:i+2) = [.9999 .9 .9999];
        end
    elseif strcmp(CopulaSpec.corrspec,'static') == 1
        lower = -1;
        upper = 1;
    end
    defvals=[];
elseif strcmp(type,'Gumbel') == 1
    if strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1
        display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
        defvals = 2*ones(ParSize*2,1);
        for i=1:2:(ParSize*2)
            lower(i:i+1) = [0+1e-6 0+1e-6];
        end
        for i=1:2:(ParSize*2)
            upper(i:i+1) = [.9999 .9999];
        end
    elseif  strcmp(CopulaSpec.corrspec,'ADCC') == 1
        display('the copula parameters is a .5*N*(N-1)*2 x 1 vector');
        defvals = 2*ones(ParSize*3,1);
        for i=1:3:(ParSize*3)
            lower(i:i+2) = [0+1e-6 0+1e-6 0+1e-6];
        end
        for i=1:3:(ParSize*3)
            upper(i:i+2) = [.9999 .9 .9999];
        end
    elseif strcmp(CopulaSpec.corrspec,'Patton') == 1
        display('the copula parameters is a .5*N*(N-1)x1 vector');
        defvals=.15*ones(ParSize,1);
        lower = [];
        upper =[];
    elseif strcmp(CopulaSpec.corrspec,'static') == 1
        lower = repmat(1.05,ParSize,1);
        upper = 20;
        defvals = repmat(2,ParSize,1);
    end
end