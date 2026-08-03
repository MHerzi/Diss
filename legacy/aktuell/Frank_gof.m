function [P,kappa4] = Frank_gof(data)
% Goodness-of-fit test für die Frank-Copula
% Input: Ein Tx2 Datenvektor (Returnzeitreihe)
% Output: P-value
%         Parameter der Placket-Copula geschätzt mit empirischer
%         Marginalverteilung
% Autor: Martin Grziska

% Anzahl der Simulationens
N=1000;
% Berechne empirische Marginalverteilung
[t,k]=size(data);
U = empiricalCopula(data);
V = empiricalCDF(data);

% Schätzen der Frank-Copula
options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6);
lower = 0.0001;
theta0 = 1;
% [ kappa4 LL4] = fmincon('frankCL',theta0,[],[],[],[],lower,[],[],options,V);
[ kappa4 LL4] = fminunc('frankCL',theta0,options,V);

% generiere Zufallsvariablen U der Frank-Copula
[uFrank,Holder] = frank_rnd(kappa4,t,1);

% Approximiere parametrische Copula
B = empiricalCopula(uFrank);

% Approximiere S
S = sum((U - B).^2);

% generiere Zufallsvariablen U der Frank-Copula
for i=1:2:N
[uFrank2(:,i:i+1),Holder] = frank_rnd(kappa4,t,i);
end

% Berechne empirische Copula der Pseudovariablen
k=1;
for i=1:2:N
U3(:,k)=empiricalCopula(uFrank2(:,i:i+1));
k=k+1;
end
% Bereche Frank-Copula für die Pseudovariablen
lower = 0.0001;
theta0 = 1;
for i=1:2:N
% [ kappaU3(i) LLU3] = fmincon('frankCL',theta0,[],[],[],[],lower,[],[],options,uFrank2(:,i:i+1));
[ kappaU3(i) LLU3] = fminunc('frankCL',theta0,options,uFrank2(:,i:i+1));
end
% generiere Zufallsvariablen U der Frank-Copula
for i=1:2:N
[uFrank3(:,i:i+1),l3] = frank_rnd(kappaU3(i),t,1);
end
% Approximiere parametrische Verteilung
k=1;
for i=1:2:N
B3(:,k)= empiricalCopula(uFrank3(:,i:i+1));
k=k+1;
end

% Approximiere S
S2 = sum((U3- B3).^2);

for i=1:size(S2,2)
count(i)=S2(i)>S(1);
end

P = sum(count)./(N/2);