function [logL, Rt1, likelihoods, Qt1] = ADCC_full_likelihood_forecast(parameters, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ, dccG)
% PURPOSE:
%        Hier wird der Ein-Perioden forecast für die Korrelationsmatrix Rt
%        bestimt
%        Restricted likelihood for use in the ADCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC parameters

%
% USAGE:
%        [logL, Rt, likelihoods, Qt] = ADCC_full_likelihood(params, stdresud, P, Q)
%
% INPUTS:
%    params      - A 2*P+Q by 1 vector of parameters of the form [dccPparameters; dccQparameters, adccPparameters]
%    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
%    garchtype:  Model        garchtype-number
%                EGARCH       0
%                GARCH        1
%                TGARCH       2
%                AVGARCH      3
%                NAGRCH       4
%                NAGARCH      5
%                APGARCH      6
%                ALLGARCH     7
%                GJRGARCH     8
%    errortype: 'NORMAL'
%               'STUDENTST'
%               'GED'
% OUTPUTS:
%    logL        - The calculated Quasi-Likelihood
%    Rt1         - a k x k x t 3 dimesnaional array of conditional
%                  correlations forecast (one period ahead)
%    likelihoods - a t by 1 vector of quasi likelihoods

%
%
% COMMENTS:
%
%
% Author: Martin Grziska
% based on the DCC code by Kevin Sheppard
% letzte Modifikation: 15. Sep 2009

[t,k]=size(data);
index=1;
H=zeros(size(data));
stdEstimate =  std(data,1);
P=dccP;
Q=dccQ;

for i=1:k
    if errortype(i) ~= 1
        gamma(i)=1;
    else gamma(i)=0;
    end
end

for i=1:k
    univariateparameters=parameters(index:index+archP(i)+o(i)+garchQ(i)+gamma(i));
    if garchtype(:,i)==0;
        H(:,i) = egarchcore(data(:,i), univariateparameters(1:end-gamma(i)), stdEstimate(:,i), archP(i), o(i), garchQ(:,i) ,1 , t);
    else
        [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparameters, data(:,i), archP(i), o(i), garchQ(i), garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+gamma(i)+1;
end

stdresid=data./sqrt(H);

params=parameters;
a          = params(index:index+dccP-1);
a_negative = params(index+dccP:index+dccP+dccG-1);
b = params(index+dccP+dccG:index+dccP+dccG+dccQ-1);
sumA = eye(k)*sum(a);
sumA_negative = eye(k)*sum(a_negative);
sumB = eye(k)*sum(b);

Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

m = max(dccP,dccQ);
Qt1 = zeros(k,k,t+m);
Rt1 = zeros(k,k,t+m);
Qt1(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt1(:,:,1:m) = repmat(Qbar,[1 1 m]);
H=[zeros(m,k);H];
logL = 0;
likelihoods = zeros(1,t+m);

stdresid = [zeros(m,k); stdresid];
negativeStdresid = stdresid .* (stdresid < 0);
violatedPSD = 0;
Qinitial = Qbar * (eye(k) - sumA - sumB) - Nbar_negative * sumA_negative;
% Für den Forecast Qt1, nehme die Residuen aus t; j fängt generell bei 2
% an, da bei j=1, die unconditional Terme stehen
for j = (m+1):t+m+1
    Qt1(:,:,j) = Qinitial;
    for i=1:P
        Qt1(:,:,j) = Qt1(:,:,j) + a(i)*(stdresid(j-i,:)'*stdresid(j-i,:));
        Qt1(:,:,j) = Qt1(:,:,j) + a_negative(i)*(negativeStdresid(j-i,:)'*negativeStdresid(j-i,:));
    end
    for i = 1:Q
        Qt1(:,:,j) = Qt1(:,:,j) + b(i)*Qt1(:,:,j-i);
    end
    Rtemp = Qt1(:,:,j)./(sqrt(diag(Qt1(:,:,j)))*sqrt(diag(Qt1(:,:,j)))');
    Rtemp = Rtemp - diag(diag(Rtemp)) + eye(k);
    Rt1(:,:,j) = Rtemp;
    maxmax = max(max(Rt1(:,:,j)));
    minmin = min(min(Rt1(:,:,j)));
    if maxmax > 1 || minmin < -1
        violatedPSD = 1;
    end
    likelihoods(j)=k*log(2*pi)+sum(log(H(j-1,:)))+log(det(Rt1(:,:,j)))+stdresid(j-1,:)*inv(Rt1(:,:,j))*stdresid(j-1,:)';
    logL = logL + likelihoods(j);
end;

Qt1 = Qt1(:,:,(m+2:t+m+1));
Rt1 = Rt1(:,:,(m+2:t+m+1));
logL = (1/2)*logL;
likelihoods = [(1/2)*likelihoods(m+1:t+m)]';

