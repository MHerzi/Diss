% Berechnet verschiedene Copulas für bivariate Datensätze, wobei U vorher
% bestimmt wurde
% u=ibm_ccola_rets(:,1);
% v=ibm_ccola_rets(:,2);
T = length(u)

% estimating some copula models

options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6);


% 1. Normal Copula
kappa1_emp = corrcoef12(norminv(u),norminv(v));
LL1_emp = NormalCopula_CL(kappa1,[u,v]);	   

% 2. Clayton's copula
lower = 0.0001;
theta0 = 1;
[ kappa2_emp LL2_emp] = fmincon('claytonCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);

% 3. Rotated Clayton copula 
lower = 0.0001;
theta0 = 1;
[ kappa3_emp LL3_emp] = fmincon('claytonCL',theta0,[],[],[],[],lower,[],[],options,1-[u,v]);

% 4. Plackett copula
lower = 0.0001;
theta0 = 1;
[ kappa4_emp LL4_emp] = fmincon('plackettCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);
% LL5 = -3.2721

% 5. Frank copula
theta0 = 1;
% [ kappa5_emp LL5_emp] = fmincon('frankCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);
[ kappa5_2_emp LL5_2_emp] = fminunc('frankCL',theta0,options,[u,v]);
% [ kappa5 LL5] = fminunc('frankCL2',theta0,options,[u,v]);

% 6. Gumbel copula
lower = 1.1;
theta0 = 2;
[ kappa6_emp LL6_emp] = fmincon('gumbelCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);

% 7. Rotated Gumbel copula
lower = 1.1;
theta0 = 2;
[ kappa7_emp LL7_emp] = fmincon('gumbelCL',theta0,[],[],[],[],lower,[],[],options,1-[u,v]);

% 8. Student's t copula
lower = [-0.9 , 2.1 ];
upper = [ 0.9 , 100 ];
theta0 = [kappa1;10];
[ kappa8_emp LL8_emp] = fmincon('tcopulaCL',theta0,[],[],[],[],lower,upper,[],options,[u,v]);

% 9. Symmetrised Joe-Clayton copula
lower = [0 , 0 ];
upper = [ 1 , 1];
theta0 = [0.25;0.25];
[ kappa9_emp LL9_emp] = fmincon('sym_jc_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v]);

LL = [LL1;LL2;LL3;LL4;LL5;LL6;LL7;LL8;LL9];
[(1:length(LL))',LL]
sortrows([(1:length(LL))',LL],2)

opt_copula = find(LL==min(LL))


% tail dependence implied by each of these copulas
tauLU = nines(9,2);
tauLU(1,:) = [0,0];                 % Normal copula has zero tail dependence
tauLU(2,:) = [2^(-1/kappa2_emp),0];     % Clayton copula has zero upper tail dependence
tauLU(3,:) = [0,2^(-1/kappa3_emp)];     % Rotated Clayton copula has zero lower tail dependence
tauLU(4,:) = [0,0];                 % Plackett copula has zero tail dependence
tauLU(5,:) = [0,0];                 % Frank copula has zero tail dependence
tauLU(6,:) = [0,2-2^(1/kappa6_emp)];    % Gumbel copula has zero lower tail dependence
tauLU(7,:) = [2-2^(1/kappa7_emp),0];    % Rotated Gumbel copula has zero upper tail dependence
tauLU(8,:) = ones(1,2)*2*tdis_cdf(-sqrt((kappa8_emp(2)+1)*(1-kappa8_emp(1))/(1+kappa8_emp(1))),kappa8_emp(2)+1);  % Student's t copula has symmetric tail dependence
tauLU(9,:) = kappa9_emp([2,1])';               % SJC copula parameters are the tail dependence coefficients, but in reverse order.
tauLU

sortrows([(1:9)',LL,tauLU],2)


% Now taking a look at a couple of time-varying copulas

% 10. Time-varying normal Copula
lower = -5*ones(3,1);  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper = 5*ones(3,1);
theta0 = [log((1+kappa1_emp)/(1-kappa1_emp));0;0];
[ kappa10_emp LL10_emp] = fmincon('bivnorm_tvp1_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa1_emp);
[LL10_emp, rho10_emp] = bivnorm_tvp1_CL(kappa10_emp,[u,v],kappa1_emp);
figure(20),plot((1:T)',rho10_emp,(1:T)',kappa1*ones(T,1),'r--'),legend('time-varying','constant'),title('Normal copula');

% 11. Time-varying rotated Gumbel copula
lower = -5*ones(3,1);  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper =  5*ones(3,1);
theta0 = [sqrt(kappa7_emp-1);0;0];
[ kappa11_emp LL11_emp] = fmincon('Gumbel_tvp1_CL',theta0,[],[],[],[],lower,upper,[],options,[1-u,1-v],kappa7_emp);
[LL11, rho11] = Gumbel_tvp1_CL(kappa11,[1-u,1-v],kappa7);
figure(21),plot((1:T)',rho11,(1:T)',kappa7*ones(T,1),'r--'),legend('time-varying','constant'),title('Rotated Gumbel copula');

% 12. Time-varying SJC copula
lower = -25*ones(6,1);  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper =  25*ones(6,1);
theta0 = [log(kappa9_emp(1)/(1-kappa9_emp(1)));0;0;log(kappa9_emp(2)/(1-kappa9_emp(2)));0;0];
[ kappa12_emp LL12_emp] = fmincon('sym_jc_tvp_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa9_emp);
[ LL12_emp tauL12_emp tauU12_emp] = sym_jc_tvp_CL(kappa12_emp,[u,v],kappa9_emp);
figure(22),subplot(2,1,1),plot((1:T)',tauL12_emp,(1:T)',kappa9_emp(1)*ones(T,1),'r--'),legend('time-varying','constant'),title('SJC copula - lower tail'),axis([0,T,0,0.8]);
subplot(2,1,2),plot((1:T)',tauU12 ,(1:T)',kappa9_emp(2)*ones(T,1),'r--'),legend('time-varying','constant'),title('SJC copula - upper tail'),axis([0,T,0,0.8]);

% 13. Time-varying t-copula
lower = -25*ones(6,1);
upper = 25*ones(6,1);
theta0 = [log((1+kappa8_emp(1))/(1-kappa8_emp(1)));0;0;kappa8_emp(2)];
[ kappa13_emp LL13_emp] = fmincon('bivt_tvp1_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa1);
% [ kappa13 LL13] = fminunc('bivt_tvp1_CL',theta0,options,[u,v],kappa1); %ohne constraints, die LLF gleicht sich aber mit der von fmincon
[LL13_emp rho13_emp] = bivt_tvp1_CL(kappa13_emp,[u,v],kappa1_emp);
figure(13),plot((1:T)',rho13_emp,(1:T)',kappa8_mp(1)*ones(T,1),'r--'),legend('time-varying','constant'),title('Student-t copula');

% % 14.Time-varying Frank copula Patton Specificaion
theta0 = [sqrt(kappa5_2-1);0;0];
% [ kappa14 LL14] = fmincon('bivfrank_tvp1_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa5_2);
[ kappa14_emp LL14_emp] =fminunc('bivfrank_tvp1_CL_neu_patton',theta0,[],[u,v],kappa5_2_emp);
[LL14_emp rho14_emp] = bivfrank_tvp1_CL_neu_patton(kappa14_emp,[u,v],kappa5_2);
figure(24),plot((1:T)',rho14,(1:T)',kappa5_mp*ones(T,1),'r--'),legend('time-varying','constant'),title('Frank copula');

% % 14a.Time-varying Frank copula Kim Specificaion
% theta0_kim = [sqrt(kappa5-1);0;0;0;0;0;0];
% [ kappa14 LL14] =fminunc('bivfrank_tvp1_CL_kim',theta0_kim,[],[u,v],kappa5);
% [LL14 rho14] = bivfrank_tvp1_CL_kim(kappa14,[u,v],kappa5);


% 15.Time-varying Clayton copula



% 16.Time-varying Gumbel Copula
lower = -5*ones(3,1);  % in theory there are no constraints, but setting loose constraints sometimes helps in the numerical optimisation
upper =  5*ones(3,1);
theta0 = [sqrt(kappa6_emp-1);0;0];
[ kappa16_emp LL16_emp] = fmincon('Gumbel_tvp1_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa6_emp);
[LL16_emp, rho16_emp] = Gumbel_tvp1_CL(kappa16_emp,[u,v],kappa6_emp);
figure(16),plot((1:T)',rho16_emp,(1:T)',kappa6_emp*ones(T,1),'r--'),legend('time-varying','constant'),title('Gumbel copula');

% 17.Time-varying Plackett Copula
lower = 0.0001;
theta0 = 1;
[ kappa17_emp LL17_emp] = fmincon('Plackett_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v],kappa4);

LL = [LL1_emp;LL2_emp;LL3_emp;LL4_emp;LL5_emp;LL6_emp;LL7_emp;LL8_emp;LL9_emp;LL10_emp;LL11_mp;LL12_emp];
[(1:length(LL))',LL]
sortrows([(1:length(LL))',LL],2)

params = [ones(7,1);2;2;3;3;6];  % number of parameters in each model
AIC = 2*LL + 2/T*params;
BIC = 2*LL + log(T)/T*params;
[(1:length(LL))',LL,AIC,BIC]
sortrows([(1:length(LL))',LL,AIC,BIC],2)
sortrows([(1:length(LL))',LL,AIC,BIC],3)
sortrows([(1:length(LL))',LL,AIC,BIC],4)
% rankings by AIC and BIC are the same as by log-likelihood (T is so large
% that k=1 vs k=6 does not impose a very large penalty)

