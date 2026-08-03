function [logL, Rt1, likelihoods, Qt1]=AGDCC_full_likelihood_forecast(parameters, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ, dccG)

% PURPOSE:
%        Restricted likelihood for use in the ADCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC parameters
% 
% USAGE:
%        [logL, Rt, likelihoods, Qt] = ADCC_full_likelihood(parameters, stdresud, P, Q)
% 
% INPUTS:
%    parameters      - A 2*P+Q by 1 vector of parameters of the form [dccPparameters; dccQparameters, adccPparameters]
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
%    errortype: 'NORMAL' = 0;     
%               'STUDENTST' = 1;    
%               'GED' = 1;
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
        H(:,i) = egarchcore_grm(data(:,i), univariateparameters(1:end-gamma(i)), stdEstimate(:,i), archP(i), o(i), garchQ(:,i), m , t);
    else
    [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparameters, data(:,i), archP(i), o(i), garchQ(i), garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+gamma(i)+1;
end

stdresid=data./sqrt(H);

a          = diag(parameters(index:index+P*k-1));       % ARCH for all residuals
a_negative = diag(parameters(index+(P*k):index+2*(P*k)-1));   % ARCH for negative residuals
b = diag(parameters(index+2*(P*k):index+2*(P*k)+(Q*k)-1));        % GARCH

% First compute Qbar, the correlation matrix of the standardized residuals,
% and Nbar, the correlation matrix of the standardized residuals,
% conditional on the standardized residuals being negative
Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

% Next compute Qt
m = max(P,Q);
Qt1 = zeros(k,k,t+m);
Rt1 = zeros(k,k,t+m);
Qt1(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt1(:,:,1:m) = repmat(Qbar,[1 1 m]);
logL = 0;
likelihoods = zeros(1,t+m);

stdresid = [zeros(m,k); stdresid];
negativeStdresid = stdresid .* (stdresid < 0);
violatedPSD = 0;
Qinitial = Qbar-a'*Qbar*a-b'*Qbar*b-a_negative'*Nbar_negative*a_negative;

for j = (m+1):t+m+1
   Qt1(:,:,j) = Qinitial;
   for i=1:P
     Qt1(:,:,j) = Qt1(:,:,j) + a'*(stdresid(j-i,:)'*stdresid(j-i,:))*a;
     Qt1(:,:,j) = Qt1(:,:,j) + a_negative'*(negativeStdresid(j-i,:)'*negativeStdresid(j-i,:))*a_negative;
   end
   for i = 1:Q
      Qt1(:,:,j) = Qt1(:,:,j) + b'*Qt1(:,:,j-i)*b;
   end
   Rtemp = Qt1(:,:,j)./(sqrt(diag(Qt1(:,:,j)))*sqrt(diag(Qt1(:,:,j)))');
   Rtemp = Rtemp - diag(diag(Rtemp)) + eye(k);
   Rt1(:,:,j) = Rtemp;
   maxmax = max(max(Rt1(:,:,j)));
   minmin = min(min(Rt1(:,:,j)));
   if maxmax > 1 || minmin < -1
       violatedPSD = 1;
   end
   likelihoods(j) = log(det(Rt1(:,:,j))) + stdresid(j-P,:)*inv(Rt1(:,:,j))*stdresid(j-P,:)';
   logL = logL + likelihoods(j);
end;

Qt1 = Qt1(:,:,(m+2:t+m+1));
Rt1 = Rt1(:,:,(m+2:t+m+1));
logL = (1/2)*logL;
likelihoods = (1/2)*likelihoods(m+1:t+m);

if ~isreal(logL) || violatedPSD 
%    disp('Imag')
%    parameters
   logL=10E+15;
end

if isinf(logL)
%    disp('Inf')
%    parameters
   logL=10E+15;
end
