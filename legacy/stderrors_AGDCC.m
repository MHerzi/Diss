function [Tstat, stderrors] = stderrors_AGDCC(data, parameters, archP, garchQ, dccP, dccQ, dccG, asymmG, garchtype, errortype, univariate)
% Computes t stat and Bollerslev-Wooldridge stderrors for AGDCC model
% Author: Martin Grziska 02/25/2010


% Berechne die Matrix A für die Bollerslev-Wooldridge Standardfehler. Hier
% darf nicht die invertierte Hesse-Matrix verwendet werden
A=zeros(length(parameters),length(parameters));
index=1;
[t,k] =size(data);
for i=1:k
    workingsize=size(univariate{i}.stderrors,2);
    A(index:index+workingsize-1,index:index+workingsize-1)=univariate{i}.stderrors^(-1); %invertiere die invertierte Hesse-Matrix
    index=index+workingsize;
end

% Berechnung der Hesse-Matrix für die AGDCC-Standardfehler
otherA=dcc_hessian('agdcc_mvgarch_full_likelihood_asymmetric_univariate',parameters, (dccP+dccQ+dccG)*k, data , archP, garchQ, dccP, dccQ, dccG, asymmG, garchtype, errortype);

% Komplette A-Matrix bilden (für die Berechnung der kompletten
% Standardfehler)
A(length(parameters)-(dccP+dccQ+dccG)*k+1:length(parameters),:) = otherA;

% Scores der univariaten GARCH zusammenfassen
% Die scores die mit Matlab garchfit geschätzt wurden, haben folgende Form:
% K, GARCH, ARCH, Leverage
jointscores=zeros(t,length(parameters));
index=1;
for i=1:k
    jointscores(:,index:index+archP(i)+garchQ(i)+asymmG(i)) = univariate{i}.scores;
    index=index+archP(i)+garchQ(i)+asymmG(i)+1;
end

% Berechnung der AGDCC-scores
h=max(abs(parameters/2),1e-2)*eps^(1/3);
hplus=parameters+h;
hminus=parameters-h;
likelihoodsplus=zeros(t,length(parameters));
likelihoodsminus=zeros(t,length(parameters));
for i=length(parameters)-(dccP+dccQ+dccG)*k+1:length(parameters)
    hparameters=parameters;
    hparameters(i)=hplus(i);
    [HOLDER, HOLDER1, indivlike] = agdcc_mvgarch_full_likelihood_asymmetric_univariate(hparameters, data , archP, garchQ, dccP, dccQ, dccG, asymmG, garchtype, errortype);
    likelihoodsplus(:,i)=indivlike;
end
for i=length(parameters)-(dccP+dccQ+dccG)*k+1:length(parameters)
    hparameters=parameters;
    hparameters(i)=hminus(i);
    [HOLDER, HOLDER1, indivlike] = agdcc_mvgarch_full_likelihood_asymmetric_univariate(hparameters, data , archP, garchQ, dccP, dccQ, dccG, asymmG, garchtype, errortype);
    likelihoodsminus(:,i)=indivlike;
end
AGDCCscores=(likelihoodsplus(:,length(parameters)-(dccP+dccQ+dccG)*k+1:length(parameters))-likelihoodsminus(:,length(parameters)-(dccP+dccQ+dccG)*k+1:length(parameters)))...
    ./(2*repmat(h(length(parameters)-(dccP+dccQ+dccG)*k+1:length(parameters))',t,1));
jointscores(:,length(parameters)-(dccP+dccQ+dccG)*k+1:length(parameters))=AGDCCscores;
B=cov(jointscores);
A=A/t;

% Berechnung der Bollerslev-Wooldridge Standardfehler
stderrors=A^(-1)*B*A^(-1)*t^(-1);
% end
load tempHt
% T-Statistik
Tstat = parameters./diag(sqrt(abs(stderrors)));