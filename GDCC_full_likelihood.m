function [logL, Rt, likelihoods, Qt] = GDCC_full_likelihood(params, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ)
% PURPOSE:
%        Restricted likelihood for use in the AGDCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC params
% 
% USAGE:
%        [logL, Rt, likelihoods, Qt] = GDCC_full_likelihood(params, data,
%        archP, garchQ, o, garchtype, errortype, dccP, dccQ)
% 
% INPUTS:
%    params      - A 2*P+Q by 1 vector of params of the form [dccPparams; dccQparams, adccPparams]
%    data        - A matrix, t x k of return series
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
% 
% OUTPUTS:
%    logL        - The calculated Quasi-Likelihood
%    Rt          - a k x k x t 3 dimesnaional array of conditional correlations
%    likelihoods - a t by 1 vector of quasi likelihoods

% 
% 
% COMMENTS:
% 
% 
% Author: Christoph Schleicher, Martin Grziska, 02/26/2010
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
    univariateparams=params(index:index+archP(i)+o(i)+garchQ(i)+gamma(i));
    if garchtype(:,i)==0;
        H(:,i) = egarchcore(data(:,i), univariateparams(1:end-gamma(i)), stdEstimate(:,i), archP(i), o(i), garchQ(i), m , t);
    else
    [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparams, data(:,i), archP(i), o(i), garchQ(i), garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+gamma(i)+1;
end

stdresid=data./sqrt(H);

[t,k] = size(stdresid);
a          = diag(params(index:index+P*k-1));       % ARCH for all residuals
b = diag(params(index+(P*k):index+(P*k)+(Q*k)-1));        % GARCH

% First compute Qbar, the correlation matrix of the standardized residuals,
% and Nbar, the correlation matrix of the standardized residuals,
% conditional on the standardized residuals being negative
Qbar = cov(stdresid);

% Next compute Qt
m = max(P,Q);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
logL = 0;
likelihoods = zeros(1,t+m);

stdresid = [zeros(m,k); stdresid];
H=[zeros(m,k);H];
Qinitial = Qbar-a'*Qbar*a-b'*Qbar*b;
for j = (m+1):t+m
   Qt(:,:,j) = Qinitial;
   for i=1:P
     Qt(:,:,j) = Qt(:,:,j) + a'*(stdresid(j-i,:)'*stdresid(j-i,:))*a;
   end
   for i = 1:Q
      Qt(:,:,j) = Qt(:,:,j) + b'*Qt(:,:,j-i)*b;
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

