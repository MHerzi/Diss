function [params, negativeLikelihood, exitFlag] = Est_AGDCC_cop_start(data,dccP,dccQ)

% Estimation of the AGDCC-model of Cappiello et al (2006)
% Inputs:
%         data: a t x k matrix of standardized residuals
%         dccP: # of ARCH lags in AGDCC model
%         dccQ: # of GARCH lags in AGDCC model
%
% Outputs: params: 2*dccP*k + dccQ*k vector of AGDCC parameters
%          negativelikelihood: negative LogL
%          exitFlag: skalar, if > 0; successfull optimization
%
% Author: Martin Grziska, 02/25/2010

[t,k] = size(data);

if exist('startingValues','var')
    agdccstarting = startingValues.correlation;
else
    [parameters_start] = Est_ADCC(data, dccP, dccQ);
    parameters_start = abs(parameters_start);
    agdccstarting = [ones(k,dccP)*parameters_start(1)/dccP;  ones(k,dccP)*parameters_start(2)/dccP; ones(k,dccQ)*parameters_start(3)/dccQ];
end
agdccstarting_2 = [ones(k,dccP)*.001/dccP;  zeros(k,dccP); ones(k,dccQ)*.001/dccQ];

options  =  optimset('fmincon');
options  =  optimset(options , 'Display'     , 'iter');
options  =  optimset(options , 'Diagnostics' , 'off');
options  =  optimset(options , 'LevenbergMarquardt' , 'on');
options  =  optimset(options , 'LargeScale'  , 'off');

A=[];
b=[];
% Constraints
% alpha>0, beta>0, gamma>0 (constriants für Cappiello et al (2006), a^2,und alpha<1, beta<1, gamma<1
LB = zeros(1,size(agdccstarting,1))+1e-6;
UB = ones(1,size(agdccstarting,1))-1e-6;

epsilon = 10^(-3);
exitFlag = 0;
count = 0;
while exitFlag < 1 && count <= 1
    if epsilon < 10^(-3)
        disp(['epsilon = ', num2str(epsilon)])
    end
    %     agdccparameter: A (alpha), G(negative) ,B (beta)
    [params, negativeLikelihood, exitFlag] = fmincon('AGDCC_likelihood_grm', agdccstarting, A, b, ...
        [], [], LB, UB, 'AGDCC_nonlincon', options, data, dccP, dccQ, epsilon);
     if negativeLikelihood == 10E+15; %10E+15 ist default likelihood-Wert; selbst wenn Optimum gefunden wird, mache neue Iteration
        exitFlag=0;
    end
    if count == 1
        agdccstarting = agdccstarting_2;
    else
        epsilon = epsilon/2;
    end
    count = count + 1;
end

% if no solution with active-set, try interior-point
if exitFlag < 1
    options = optimset(options, 'Algorithm','interior-point');
    epsilon = 10^(-3);
    exitFlag = 0;
    count = 0;
    while exitFlag < 1 && count <= 1
        if epsilon < 10^(-3)
            disp(['epsilon = ', num2str(epsilon)])
        end
        %     agdccparameter: A (alpha), G(negative) ,B (beta)
        [params, negativeLikelihood, exitFlag] = fmincon('AGDCC_likelihood_grm', agdccstarting, A, b, ...
            [], [], LB, UB, 'AGDCC_nonlincon', options, data, dccP, dccQ, epsilon);
        if count == 1
            agdccstarting = agdccstarting_2;
        else
            epsilon = epsilon/2;
        end
        count = count + 1;
    end
end

if exitFlag < 0
    disp('optimization did not finish successfully')
    keyboard
end
