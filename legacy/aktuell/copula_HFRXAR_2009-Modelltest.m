function [optCopula_HFRXAR,opttheta] = copula_HFRXAR_2009_Modelltest()
% % Berechnet verschiedene Copulas für bivariate Datensätze, wobei U vorher
% % bestimmt wurde
cd G:\Dissertation\Daten
load('V_DJW1');
load('V_HFRXAR');
u = u_HFRXAR(3:end,:);
v = u_DJW1;
T = length(u);

% estimating some copula models

options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6,'HessUpdate','bfgs');


% 1. Normal Copula
kappa1 = corrcoef12(norminv(u),norminv(v));
LL1 = NormalCopula_CL(kappa1,[u,v]);	   

% 2. Clayton's copula
lower = -1.0001;
theta0 = 1;
[ kappa2 LL2] = fmincon('claytonCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);

% 3. Rotated Clayton copula 
lower = 0.0001;
theta0 = 1;
[ kappa3 LL3] = fmincon('claytonCL',theta0,[],[],[],[],lower,[],[],options,1-[u,v]);

% 4. Plackett copula
lower = 0.0001;
theta0 = 1;
[ kappa4 LL4] = fmincon('plackettCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);

% 5. Frank copula
theta0 = 3;
% [ kappa5 LL5] = fmincon('frankCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);
[ kappa5_2 LL5_2] = fminunc('frankCL',theta0,options,[u,v]);
% [ kappa5 LL5] = fminunc('frankCL2',theta0,options,[u,v]);

% 6. Gumbel copula
lower = 1.1;
theta0 = 2;
[ kappa6 LL6] = fmincon('gumbelCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);

% 7. Rotated Gumbel copula
lower = 1.1;
theta0 = 2;
[ kappa7 LL7] = fmincon('gumbelCL',theta0,[],[],[],[],lower,[],[],options,1-[u,v]);

% 8. Student's t copula
lower = [-0.9 , 2.1 ];
upper = [ 0.9 , 100 ];
theta0 = [kappa1;10];
[ kappa8 LL8] = fmincon('tcopulaCL',theta0,[],[],[],[],lower,upper,[],options,[u,v]);

% 9. Symmetrised Joe-Clayton copula
lower = [0 , 0 ];
upper = [ 1 , 1];
theta0 = [0.25;0.25];
[ kappa9 LL9] = fmincon('sym_jc_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v]);

LL = [LL1;LL2;LL3;LL4;LL5_2;LL6;LL7;LL8;LL9];
[(1:length(LL))',LL]
sortrows([(1:length(LL))',LL],2)

opt_copula = find(LL==min(LL))


% tail dependence implied by each of these copulas
tauLU = nines(9,2);
tauLU(1,:) = [0,0];                 % Normal copula has zero tail dependence
tauLU(2,:) = [2^(-1/kappa2),0];     % Clayton copula has zero upper tail dependence
tauLU(3,:) = [0,2^(-1/kappa3)];     % Rotated Clayton copula has zero lower tail dependence
tauLU(4,:) = [0,0];                 % Plackett copula has zero tail dependence
tauLU(5,:) = [0,0];                 % Frank copula has zero tail dependence
tauLU(6,:) = [0,2-2^(1/kappa6)];    % Gumbel copula has zero lower tail dependence
tauLU(7,:) = [2-2^(1/kappa7),0];    % Rotated Gumbel copula has zero upper tail dependence
tauLU(8,:) = ones(1,2)*2*tdis_cdf(-sqrt((kappa8(2)+1)*(1-kappa8(1))/(1+kappa8(1))),kappa8(2)+1);  % Student's t copula has symmetric tail dependence
tauLU(9,:) = kappa9([2,1])';               % SJC copula parameters are the tail dependence coefficients, but in reverse order.
tauLU

sortrows([(1:9)',LL,tauLU],2)


% Now taking a look at a couple of time-varying copulas

% 10. Time-varying normal Copula
lower = -5*ones(3,1);  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper = 5*ones(3,1);
theta0 = [log((1+kappa1)/(1-kappa1));0;0];
theta0 = [log((1+kappa1)/(1-kappa1));0;0;0;0];
[ kappa10 LL101] = fmincon('bivnorm_tvp1_CL_arma22',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa1);
[LL10, rho10] = bivnorm_tvp1_CL(kappa10,[u,v],kappa1);
figure(10),plot((1:T)',rho10,(1:T)',kappa1*ones(T,1),'r--'),legend('time-varying','constant'),title('Normal copula');

% 11. Time-varying rotated Gumbel copula
lower = -5*ones(3,1);  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper =  5*ones(3,1);
theta0 = [sqrt(kappa7-1);0;0];
[ kappa11 LL11] = fmincon('Gumbel_tvp1_CL_2',theta0,[],[],[],[],lower,upper,[],options,[1-u,1-v],kappa7);
[LL11, rho11] = Gumbel_tvp1_CL(kappa11,[1-u,1-v],kappa7);
figure(11),plot((1:T)',rho11,(1:T)',kappa7*ones(T,1),'r--'),legend('time-varying','constant'),title('Rotated Gumbel copula');

% 12. Time-varying SJC copula
lower = -25*ones(6,1);  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper =  25*ones(6,1);
theta0 = [log(kappa9(1)/(1-kappa9(1)));0;0;log(kappa9(2)/(1-kappa9(2)));0;0];
[ kappa12 LL12] = fmincon('sym_jc_tvp_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa9);
[ LL12 tauL12 tauU12] = sym_jc_tvp_CL(kappa12,[u,v],kappa9);
figure(12),subplot(2,1,1),plot((1:T)',tauL12,(1:T)',kappa9(1)*ones(T,1),'r--'),legend('time-varying','constant'),title('SJC copula - lower tail'),axis([0,T,0,0.8]);
subplot(2,1,2),plot((1:T)',tauU12 ,(1:T)',kappa9(2)*ones(T,1),'r--'),legend('time-varying','constant'),title('SJC copula - upper tail'),axis([0,T,0,0.8]);

% 13. Time-varying t-copula
theta0 = [log((1+kappa8(1))/(1-kappa8(1)));0;0;kappa8(2)];
[ kappa13 LL13] = fminunc('bivt_tvp1_CL',theta0,options,[u,v],kappa1); %ohne constraints, die LLF gleicht sich aber mit der von fmincon
[LL13 rho13] = bivt_tvp1_CL(kappa13,[u,v],kappa1);
figure(13),plot((1:T)',rho13,(1:T)',kappa8(1)*ones(T,1),'r--'),legend('time-varying','constant'),title('Student-t copula');

% % 14.Time-varying Frank copula Patton Specificaion
% theta0 = [sqrt(kappa5_2-1);0;0];
% [ kappa14 LL14] = fmincon('bivfrank_tvp1_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa5_2);
k=1;
for i = 0.9:0.1:5
theta0 = [0;i;i];
[ kappa14(k,:) LL14(k,:)] =fminunc('bivfrank_tvp1_CL_5',theta0,[],[u,v],kappa5_2);
[LL14(k,:) rho14(k,:)] = bivfrank_tvp1_CL_5(kappa14(k,:),[u,v],kappa5_2);
k=k+1;
end
figure(14),plot((1:T)',rho14,(1:T)',kappa5_2*ones(T,1),'r--'),legend('time-varying','constant'),title('Frank copula');

% 15.Time-varying Clayton copula
% theta0=[1;0.3;0.3];
% lower=-1.00001;
% [kappa15, LL15] = fmincon('bivclayton_tvp1_CL',theta0,[],[],[],[],lower,[],[],options,[u,v],kappa2);
% [LL15 rho15] = bivclayton_tvp1_CL(kappa15,[u,v],kappa2);
% figure(15),plot((1:T)',rho15,(1:T)',kappa2*ones(T,1),'r--'),legend('time-varying','constant'),title('Clayton copula');
% aufgrund von Problemen bei der Iteration wird eine unbeschränkte
% Optimierung ausgeführt. Allerdings ist darauf zu achten, die Copula-Werte
% die in diesem Fall untere Grenze einhalten
theta0=[0.2;0.2;0.2];
% Es wurden mehrere Werte (also Startwerte) für theta0 eingesetzt, alle
% konvergietren jedoch zur gleichen Lösung
[kappa15, LL15] = fminunc('bivclayton_tvp1_CL',theta0,options,[u,v],kappa2);
[LL15 rho15] = bivclayton_tvp1_CL(kappa15,[u,v],kappa2);
figure(15),plot((1:T)',rho15,(1:T)',kappa2*ones(T,1),'r--'),legend('time-varying','constant'),title('Clayton copula');

% 16.Time-varying Gumbel Copula
lower = 1.1;  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper =  5*ones(3,1);
theta0 = [sqrt(kappa6-1);0;0];
[ kappa16 LL16] = fmincon('Gumbel_tvp1_CL_5',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa6);
[LL16, rho16] = Gumbel_tvp1_CL(kappa16,[u,v],kappa6);
figure(16),plot((1:T)',rho16,(1:T)',kappa6*ones(T,1),'r--'),legend('time-varying','constant'),title('Gumbel copula 5 lag');

% 17.Time-varying Plackett Copula
options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6);
lower = 0.0001;
theta0 = [10,0,0];
[ kappa17 LL17] = fmincon('PlackettCL_tvp_2',theta0,[],[],[],[],lower,[],[],options,[u,v],kappa4);
[LL17, rho17] = PlackettCL_tvp(kappa17,[u v], kappa4);
figure(17),plot((1:T)',rho17,(1:T)',kappa4*ones(T,1),'r--'),legend('time-varying','constant'),title('Plackett copula');

LL = [LL1;LL2;LL3;LL4;LL5_2;LL6;LL7;LL8;LL9;LL10;LL11;LL12;LL13];
[(1:length(LL))',LL]
sortrows([(1:length(LL))',LL],2)

params = [ones(7,1);2;2;3;3;6;3];  % number of parameters in each model
AIC = 2*LL + 2/T*params;
BIC = 2*LL + log(T)/T*params;
[(1:length(LL))',LL,AIC,BIC]
sortrows([(1:length(LL))',LL,AIC,BIC],2)
sortrows([(1:length(LL))',LL,AIC,BIC],3)
sortrows([(1:length(LL))',LL,AIC,BIC],4)
% rankings by AIC and BIC are the same as by log-likelihood (T is so large
% that k=1 vs k=6 does not impose a very large penalty)

% Goodness-of-fit tests für Copulas
cd F:\Dissertation\Daten
x=xlsread('Hedge_20081027_ab_20040702',1);
xprice = log(x(:,1));
xlag=lag(xprice,1);
xret=xprice-xlag;
xret=xret(2:end);
yprice = log(x(:,12));
ylag=lag(yprice,1);
yret=yprice-ylag;
yret=yret(2:end);
data=[xret yret];
disp('Goodness-of-fit Test für die Normal-Copula')
[PNormal_gof] = tv_Normal_gof(data);
disp('Goodness-of-fit Test für die Student T-Copula')
[PStudentT_gof] = tv_StudentT_gof(data);
disp('Goodness-of-fit Test für die Gumbel-Copula')
[PGumbel_gof] = tv_Gumbel_gof(data);
disp('Goodness-of-fit Test für die Frank-Copula')
[PFrank_gof] = tv_Frank_gof(data);
disp('Goodness-of-fit Test für die Clayton-Copula')
[PClayton_gof] = tv_Clayton_gof(data);
% disp('Goodness-of-fit Test für die Plackett-Copula')
% [PPlackett_gof]=tv_Plackett_gof(data,rho17);
disp('Goodness-of-fit Test für die symmetrische Joe Clayton-Copula')
 [PsymJC_gof_tvs]=sym_JC_gof_tv(data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [PJC_gof,kappa_JC] = tv_symJoeClayton_gof(data,tauL12,tauU12);

optCopula_HFRXAR = 'Plackett';
opttheta = rho17;