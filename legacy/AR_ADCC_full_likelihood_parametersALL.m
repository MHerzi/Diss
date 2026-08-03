function [logL, Rt, likelihoods, Qt] = AR_ADCC_full_likelihood_parametersALL(parameters, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ, dccG, arlag)
% PURPOSE:
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
%               'SKEWT':      4
% OUTPUTS:
%    logL        - The calculated Quasi-Likelihood
%    Rt          - a k x k x t 3 dimesnaional array of conditional correlations
%    likelihoods - a t by 1 vector of quasi likelihoods

% 
% 
% COMMENTS:
% 
% 
% Author: Martin Grziska based on code of Kevin Sheppard
% last modification: 02/26/2010
% based on the DCC code by Kevin Sheppard

[t,k]=size(data);
stdEstimate =  std(data,1);
P=dccP;
Q=dccQ;
m=max(max(archP),max(garchQ));
lagdiff=max(arlag)-1;
resid = zeros(t-lagdiff,k);

index = 1;
index_new=1;
for i=1:k
    if errortype(i)==1
        error(i)=0;
    elseif errortype(i)>1 && errortype(i)>4
        error(i)=1;
    elseif errortype(i) == 4
        error(i) = 2;
    end
end
for i=1:k
    b0 = parameters(index+archP(i)+garchQ(i)+o(i)+1:index+archP(i)+garchQ(i)+o(i)+arlag(i));
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
        parameters_new(index_new:index_new+archP(i)+garchQ(i)+o(i),1) = parameters(index:index+archP(i)+garchQ(i)+o(i));
        index = index+archP(i)+garchQ(i)+o(i)+arlag(i)+error(i)+1;
        index_new = index_new+archP(i)+garchQ(i)+o(i)+error(i)+1;
    elseif error(i)~=0
        parameters_new(index_new:index_new+archP(i)+garchQ(i)+o(i),1) = parameters(index:index+archP(i)+garchQ(i)+o(i));
        parameters_new = [parameters_new; parameters(index+archP(i)+garchQ(i)+o(i)+arlag(i)+1)];
        index = index+archP(i)+garchQ(i)+o(i)+error(i)+arlag(i)+1;
        index_new = index_new+archP(i)+garchQ(i)+o(i)+error(i)+1;
    end
end

H=zeros(size(resid));
[t,k]=size(resid);

index=1;
for i=1:k
    univariateparameters=parameters_new(index:index+archP(i)+o(i)+garchQ(i)+error(i));
    if garchtype(:,i)==0
        H(:,i) = egarchcore_grm(resid(:,i), univariateparameters(1:end-error(i)), stdEstimate(:,i), archP(i), o(i), garchQ(i), m , t);
    else
    [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparameters, resid(:,i), archP(i), o(i), garchQ(i), garchtype(i), errortype(i), stdEstimate(i));
    end
    index=index+archP(i)+garchQ(i)+o(i)+error(i)+1;
end

stdresid=resid./sqrt(H);

params=[parameters_new; parameters(end-2:end)];
a          = params(index:index+dccP-1);       
a_negative = params(index+dccP:index+dccP+dccG-1);   
b = params(index+dccP+dccG:index+dccP+dccG+dccQ-1);       
sumA = eye(k)*sum(a);
sumA_negative = eye(k)*sum(a_negative);
sumB = eye(k)*sum(b);

Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

m = max(dccP,dccQ);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
H=[zeros(m,k);H];
logL = 0;
likelihoods = zeros(1,t+m);

stdresid = [zeros(m,k); stdresid];
negativeStdresid = stdresid .* (stdresid < 0);
violatedPSD = 0;
Qinitial = Qbar * (eye(k) - sumA - sumB) - Nbar_negative * sumA_negative;
for j = (m+1):t+m
   Qt(:,:,j) = Qinitial;
   for i=1:P
     Qt(:,:,j) = Qt(:,:,j) + a(i)*(stdresid(j-i,:)'*stdresid(j-i,:));
     Qt(:,:,j) = Qt(:,:,j) + a_negative(i)*(negativeStdresid(j-i,:)'*negativeStdresid(j-i,:));
   end
   for i = 1:Q
      Qt(:,:,j) = Qt(:,:,j) + b(i)*Qt(:,:,j-i);
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

