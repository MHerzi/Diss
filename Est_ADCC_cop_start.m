function [adccparameters, negativeLikelihood, exitFlag] = Est_ADCC_cop_start(stdresid, dccP, dccQ)

options  =  optimset('fmincon');
options  =  optimset(options , 'Display'     , 'iter');
options  =  optimset(options , 'Diagnostics' , 'off');
options  =  optimset(options , 'LevenbergMarquardt' , 'off');
options  =  optimset(options , 'LargeScale'  , 'off');
options  =  optimset(options , 'TolCon'      , 1e-12);
options  =  optimset(options , 'TolX'      , 1e-12);
options  =  optimset(options , 'Algorithm'   ,'active-set');

% Schätze startwerte mit DCC
[startingvalues] = Est_DCC(stdresid, dccP, dccQ);
startingvalues=startingvalues.^2; %parameter werden in der Funktion quadriert

if exist('startingValues','var')
    adccstarting = startingValues.correlation;
else
    adccstarting = [startingvalues(1)/dccP,  ones(1,dccP)*.01, startingvalues(2)/dccQ]';
end
adccstarting_2 = [ones(1,dccP)*.001/dccP,  zeros(1,dccP), ones(1,dccQ)*.001/dccQ]';

% GJR-Constraints nach Matlab
% mit GJR-Gleichung sigma = omega + beta*sigma_t-1 + alpha*epsilon_t^2 + gamma*I*epsilon_t-1
% sind die constraints:
% 1. alpha + .5*gamma + beta <1
% 2. alpha >0
% 3. alpha+gamma>0
% 4. beta>0
% A = [ones(1,dccP) ones(1,dccP)*.5 ones(1,dccQ); -ones(1,dccP) zeros(1,dccP) zeros(1,dccQ); -ones(1,dccP) -ones(1,dccP) zeros(1,dccQ); zeros(1,dccP) zeros(1,dccP) -ones(1,dccQ) ];
% b = [1-1e-6; 0-1e-6; 0-1e-6; 0-1e-6];

% Constraints nach Cappiello et al (2006): quadrierte parameter werden
% verwendet: parameter dürfen nicht negativ werden, zusätzlich: alpha +
% .5*gamma + beta <1
% A = [-1 0 0;0 -1 0; 0 0 -1;ones(1,dccP) ones(1,dccP)*.5 ones(1,dccQ)];
% b = [zeros(3,1)-2*options.TolCon; 1-2*options.TolCon];

% Neue Contraints: a>0,g>0,b>0; a<1,g<1,b<1; die anderen Constraints sind
% nichtlinear (siehe 'ADCC_nonlincon_grm')
A=[];
b=[];
lower = zeros(3,1)+1e-6;
upper = [.99 .9 .99];

epsilon = 10^(-6);
exitFlag = 0;
count = 0;
while exitFlag < 1 && count <= 1
    if epsilon < 10^(-3)
        disp(['epsilon = ', num2str(epsilon)])
    end
    [adccparameters, negativeLikelihood, exitFlag] = fmincon('ADCC_likelihood', adccstarting, A, b, ...
        [],[],lower, upper, 'ADCC_nonlincon_grm', options, stdresid, dccP, dccQ, epsilon);
    if negativeLikelihood == 10E+15; %10E+15 ist default likelihood-Wert; selbst wenn Optimum gefunden wird, mache neue Iteration
        exitFlag=0;
    end
    if count == 1
        adccstarting = adccstarting_2;
    else
        epsilon = epsilon/2;
    end
    count = count + 1;
end

if exitFlag < 1
    options = optimset(options,'Algorithm','interior-point');
    while exitFlag < 1 && count <= 1
        if epsilon < 10^(-3)
            disp(['epsilon = ', num2str(epsilon)])
        end
        [adccparameters, negativeLikelihood, exitFlag] = fmincon('ADCC_likelihood', adccstarting, A, b, ...
            [],[],lower, upper, 'ADCC_nonlincon_grm', options, stdresid, dccP, dccQ, epsilon);
        if negativeLikelihood == 10E+15; %10E+15 ist default likelihood-Wert; selbst wenn Optimum gefunden wird, mache neue Iteration
            exitFlag=0;
        end
        if count == 1
            adccstarting = adccstarting_2;
        else
            epsilon = epsilon/2;
        end
        count = count + 1;
    end
end
