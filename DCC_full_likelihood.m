function [logL, Rt, likelihoods, Qt] = DCC_full_likelihood(parameters, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ)
% PURPOSE:
%        Restricted likelihood for use in the ADCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC parameters
%
% USAGE:
%        [logL, Rt, likelihoods, Qt] = DCC_full_likelihood(params, stdresud, P, Q)
%
% INPUTS:
%    params      - A 2*P+Q by 1 vector of parameters of the form [dccPparameters; dccQparameters, adccPparameters]
%    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
%
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
%
%    errortype: 'NORMAL':     1
%               'STUDENTST':  2
%               'GED':        3
% OUTPUTS:
%    logL        - The calculated Quasi-Likelihood
%    Rt          - a k x k x t 3 dimesnaional array of conditional correlations
%    likelihoods - a t by 1 vector of quasi likelihoods

%
%
% COMMENTS:
%
%
% Author: Martin Grziska
% based on the DCC code by Kevin Sheppard

[t,k]=size(data);
index=1;
H=zeros(size(data));
stdEstimate =  std(data,1);
P=dccP;
Q=dccQ;
m=max(max(archP),max(garchQ));

for i=1:k
    if errortype(i) ~= 1
        gamma(i)=1;
    else gamma(i)=0;
    end
end


for i=1:k
    univariateparameters=parameters(index:index+archP(i)+o(i)+garchQ(i)+gamma(i));
    if garchtype(:,i)==0;
        H(:,i) = egarchcore(data(:,i), univariateparameters(1:end-gamma(i)), stdEstimate(:,i), archP(i), o(i), garchQ(i), m , t);
    else
        %         set leverage term to zero if model contains not leverage but
        %         power-parameter
        if garchtype(i) == 4
            o(i) = 0;
        elseif garchtype(i) == 6
            %     special case: APGARCH contains leverage and power-parameter
            o(i) = 1;
        end
        
        [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparameters, data(:,i), archP(i), o(i), garchQ(i), garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+gamma(i)+1;
end

stdresid=data./sqrt(H);

params=parameters;
a          = params(index:index+dccP-1);
b = params(index+dccP:index+dccP+dccQ-1);
sumA = eye(k)*sum(a);
sumB = eye(k)*sum(b);

Qbar = cov(stdresid);

m = max(dccP,dccQ);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
H=[zeros(m,k);H];
logL = 0;
likelihoods = zeros(1,t+m);

stdresid = [zeros(m,k); stdresid];
violatedPSD = 0;
Qinitial = Qbar * (eye(k) - sumA.^2 - sumB.^2);
for j = (m+1):t+m
    Qt(:,:,j) = Qinitial;
    for i=1:P
        Qt(:,:,j) = Qt(:,:,j) + a(i)*(stdresid(j-i,:)'*stdresid(j-i,:))*a(i);
    end
    for i = 1:Q
        Qt(:,:,j) = Qt(:,:,j) + b(i)*Qt(:,:,j-i)*b(i);
    end
    Rtemp = Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
    Rtemp = Rtemp - diag(diag(Rtemp)) + eye(k);
    Rt(:,:,j) = Rtemp;
    maxmax = max(max(Rt(:,:,j)));
    minmin = min(min(Rt(:,:,j)));
    if maxmax > 1 || minmin < -1
        violatedPSD = 1;
    end
    likelihoods(j)=k*log(2*pi)+sum(log(H(j,:)))+log(det(Rt(:,:,j)))+stdresid(j,:)*inv(Rt(:,:,j))*stdresid(j,:)';
    logL = logL + likelihoods(j);
end;

Qt = Qt(:,:,(m+1:t+m));
Rt = Rt(:,:,(m+1:t+m));
logL = (1/2)*logL;
likelihoods = [(1/2)*likelihoods(m+1:t+m)]';

