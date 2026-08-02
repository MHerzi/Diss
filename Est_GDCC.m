function [params, negativeLikelihood, exitFlag] = Est_GDCC(data,dccP,dccQ)
% Estimation of the AGDCC-model of Cappiello et al (2006)
% Inputs:
%         data: a t x k matrix od standardized residuals
%         dccP: # of ARCH lags in AGDCC model
%         dccQ: # of GARCH lags in AGDCC model
%
% Outputs: params: 2*dccP*k + dccQ*k vector of AGDCC parameters
%
% Author: Martin Grziska, 03/21/2010

[t,k] = size(data);

if exist('startingValues','var')
    gdccstarting = startingValues.correlation;
else
    [gdccstarting] = Est_DCC(data,dccP,dccQ);
    gdccstarting = [ones(k,dccP)*gdccstarting(1)/dccP; ones(k,dccQ)*gdccstarting(2)/dccQ];
end
gdccstarting_2 = [ones(k,dccP)*.001/dccP; ones(k,dccQ)*.001/dccQ];

options  =  optimset('fmincon');
options  =  optimset(options , 'Display'     , 'iter');
options  =  optimset(options , 'Diagnostics' , 'off');
options  =  optimset(options , 'LevenbergMarquardt' , 'on');
options  =  optimset(options , 'LargeScale'  , 'off');
options  =  optimset(options , 'MaxIter',500,'MaxFunEvals',10000);
options  =  optimset(options , 'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6);

lower = zeros(k*2,1)+1e-6; %parameter >0
upper = ones(k*2,1)-1e-6;% parameters <1
epsilon = 10^(-3);
exitFlag = 0;
count = 0;
while exitFlag < 1 && count <= 5
    if epsilon < 10^(-3)
        disp(['epsilon = ', num2str(epsilon)])
    end
    [params, negativeLikelihood, exitFlag] = fmincon('GDCC_likelihood', gdccstarting, [], [], ...
        [], [], lower, upper, 'GDCC_nonlincon', options, data, dccP, dccQ, epsilon);
    if count == 1
        gdccstarting = gdccstarting_2;
    else
        epsilon = epsilon/2;
    end
    count = count + 1;
end

if exitFlag < 1
    options = optimset(options,'Algorithm','interior-point');
    epsilon = 10^(-3);
    exitFlag = 0;
    count = 0;
    while exitFlag < 1 && count <= 5
        if epsilon < 10^(-3)
            disp(['epsilon = ', num2str(epsilon)])
        end
        [params, negativeLikelihood, exitFlag] = fmincon('GDCC_likelihood', gdccstarting, [], [], ...
            [], [], lower, upper, 'GDCC_nonlincon', options, data, dccP, dccQ, epsilon);
        if count == 1
            gdccstarting = gdccstarting_2;
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

