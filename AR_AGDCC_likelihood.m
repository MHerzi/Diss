function [logL, Rt, likelihoods, Qt] = AR_AGDCC_likelihood(params, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ, dccG, arlag)
% PURPOSE:
%        Restricted likelihood for use in the AGDCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC params
% 
% USAGE:
%        [logL, Rt, likelihoods, Qt] = AGDCC_full_likelihood(params, stdresid, P, Q)
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
stdEstimate =  std(data,1);
P=dccP;
Q=dccQ;
G=dccG;
m=max(max(archP),max(garchQ));
lagdiff=max(arlag)-1;
resid = zeros(t-lagdiff,k);

index = 1;
index_new=1;
for i=1:k
    if errortype(i)==1
        error(i)=0;
    else
        error(i)=1;
    end
end
for i=1:k
    b0 = params(index+archP(i)+garchQ(i)+o(i)+1:index+archP(i)+garchQ(i)+o(i)+arlag(i));
    n = length(data);
    nlag = arlag(i)-1;
    x = [ones(n,1) mlag(data(:,i),nlag)];
    x = trimr(x,nlag,0);
    reg = trimr(data(:,i),nlag,0);
    reg = reg - x*b0;
    if nlag < lagdiff
    resid(:,i) = trimr(reg,lagdiff-nlag,0);
    else 
        resid(:,i) = reg;
    end
    if error(i)==0;
        params_new(index_new:index_new+archP(i)+garchQ(i)+o(i),1) = params(index:index+archP(i)+garchQ(i)+o(i));
        index = index+archP(i)+garchQ(i)+o(i)+arlag(i)+error(i)+1;
        index_new = index_new+archP(i)+garchQ(i)+o(i)+error(i)+1;
    elseif error(i)~=0
        params_new(index_new:index_new+archP(i)+garchQ(i)+o(i),1) = params(index:index+archP(i)+garchQ(i)+o(i));
        params_new = [params_new; params(index+archP(i)+garchQ(i)+o(i)+arlag(i)+1)];
        index = index+archP(i)+garchQ(i)+o(i)+error(i)+arlag(i)+1;
        index_new = index_new+archP(i)+garchQ(i)+o(i)+error(i)+1;
    end
end
H=zeros(size(resid));
[t,k]=size(resid);

index=1;
for i=1:k
    univariateparams=params_new(index:index+archP(i)+o(i)+garchQ(i)+error(i));
    if garchtype(:,i)==0
        H(:,i) = egarchcore_grm(resid(:,i), univariateparams(1:end-error(i)), stdEstimate(:,i), archP(i), o(i), garchQ(:,i), m , t);
    else
    [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparams, resid(:,i), archP(i), o(i), garchQ(i), garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+error(i)+1;
end

stdresid=resid./sqrt(H);

[t,k] = size(stdresid);
params=[params_new; params(end-(2*P*k+Q*k)+1:end)];
a          = diag(params(index:index+P*k-1));       % ARCH for all residuals
a_negative = diag(params(index+(P*k):index+2*(P*k)-1));   % ARCH for negative residuals
b = diag(params(index+2*(P*k):index+2*(P*k)+(Q*k)-1));        % GARCH

% First compute Qbar, the correlation matrix of the standardized residuals,
% and Nbar, the correlation matrix of the standardized residuals,
% conditional on the standardized residuals being negative
Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

% Next compute Qt
m = max(P,Q);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
logL = 0;
likelihoods = zeros(1,t+m);

stdresid = [zeros(m,k); stdresid];
negativeStdresid = stdresid .* (stdresid < 0);
violatedPSD = 0;
Qinitial = Qbar-a'*Qbar*a-b'*Qbar*b-a_negative'*Nbar_negative*a_negative;

for j = (m+1):t+m
   Qt(:,:,j) = Qinitial;
   for i=1:P
     Qt(:,:,j) = Qt(:,:,j) + a'*(stdresid(j-i,:)'*stdresid(j-i,:))*a;
     Qt(:,:,j) = Qt(:,:,j) + a_negative'*(negativeStdresid(j-i,:)'*negativeStdresid(j-i,:))*a_negative;
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
   
   likelihoods(j) = log(det(Rt(:,:,j))) + stdresid(j,:)*inv(Rt(:,:,j))*stdresid(j,:)';
   logL = logL + likelihoods(j);
end;

Qt = Qt(:,:,(m+1:t+m));
Rt = Rt(:,:,(m+1:t+m));
logL = (1/2)*logL;
likelihoods = (1/2)*likelihoods(m+1:t+m);

if isreal(theta) == 0 || violateSPD;
   logL = 1e6;
elseif isreal(logL) == 0;
   logL = 1e7;
elseif isnan(logL)
   logL = 1e8;
elseif isinf(logL)
   logL = 1e9;
end
