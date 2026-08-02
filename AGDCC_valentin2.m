% AGDCCC t-copula
[t,k]=size(data);
dccP=1;
dccQ=1;

% Schätzen der t-copula mit AGDCC-Struktur
% GJR-constraints (Hamilton S.669)
A = [eye(k) eye(k)*.5 eye(k)*0 zeros(k,1)]*(-1);
b = zeros(k,1) + 2*options.TolCon;

% Bounds: alle parameter > 0
ub = [ones(k,dccP)*0.99; ones(k,dccP)*0.9; ones(k,dccQ)*0.999; 50];
lb = [zeros(k*3,dccP)+2*options.TolCon; 2.1];

epsilon = 10^(-3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

options  =  optimset('fmincon');
options  =  optimset(options , 'TolFun'      , 1e-006);
options  =  optimset(options , 'Display'     , 'iter');
options  =  optimset(options , 'Diagnostics' , 'off');
options  =  optimset(options , 'LargeScale'  , 'off');
options  =  optimset(options , 'Display'     , 'off');
options  =  optimset(options , 'MaxFunEvals' , 400*(2+1+1));
% Zur Bestiimmung der Startwerte schätze univariate GARCH für jede
% Zeitreihe
for i=1:k
    [parameters{i}, likelihood, stderrors, robustSE, ht, scores] = fattailed_garch(daten(:,i) , 1 , 1 , 'NORMAL',[],options);
end

for i=1:k
    theta0(i,1)=parameters{i}(2);
end
theta0 = [theta0;repmat(.001,k,1)];
h=size(theta0,1)+1;
for i=1:k
    theta0(h) = parameters{i}(end);
    h=h+1;
end
% Startwerte
theta0 = [theta0;10];

%Schätzung 
options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6,'Algorithm','interior-point','Hessian','bfgs');
[ kappa_t_AGDCC LL_t_AGDCC] = fmincon('multi_tCopula_tvp1_AGDCCRt',theta0,A,b,[],[],lb,ub,'multi_tCopula_AGDCC_nonlincon',options,data,dccP,dccQ,epsilon);
[LL_t_AGDCC Rt_t_AGDCC] = multi_tCopula_tvp1_AGDCCRt(kappa_t_AGDCC,data,dccP,dccQ);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% 

theta0=theta0(1:end-1);
A = [eye(k) eye(k)*.5 eye(k)*0]*(-1);
b = zeros(k,1) - 2*options.TolCon;

% Bounds: alle parameter > 0
ub = [ones(k,dccP)*0.99; ones(k,dccP)*0.9; ones(k,dccQ)*0.999];
lb = zeros(k*3,dccP)+2*options.TolCon;
[ kappa_Gauss_AGDCC LL_Gauss_AGDCC] = fmincon('multi_GaussCopula_tvp1_AGDCCRt',theta0,A,b,[],[],lb,ub,'multi_GaussCopula_AGDCC_nonlincon',options,data,dccP,dccQ,epsilon);
[ LL_Gauss_AGDCC Rt_Gauss_AGDCC] = multi_GaussCopula_tvp1_AGDCCRt(kappa_Gauss_AGDCC,data,dccP,dccQ,epsilon);


