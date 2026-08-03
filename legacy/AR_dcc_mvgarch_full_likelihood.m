function [logL, Rt, likelihoods, Qt]=AR_dcc_mvgarch_full_likelihood(params, data, archP, garchQ, o, garchtype, errortype, dccP, dccQ, arlag, const)
% PURPOSE:
%        Restricted likelihood for use in the ADCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC params
%
% USAGE:
%        [logL, Rt, likelihoods, Qt] =...
%        AR_dcc_mvgarch_full_likelihood(params, data, archP, garchQ, o,...
%        garchtype, errortype, dccP, dccQ, arlag)
%
% INPUTS:
%    params      - A 2*P+Q by 1 vector of params of the form [dccPparams; dccQparams, adccPparams]
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
% Author: Martin Grziska based on code of Kevin Sheppard
% last modification: 02/26/2010
% based on the DCC code by Kevin Sheppard

[t,k]=size(data);
stdEstimate =  std(data,1);
P=dccP;
Q=dccQ;
m=max(max(archP),max(garchQ));
lagdiff=max(arlag); %falls es unterschiedliche laglängen gibt müssen diese später korrigiert werden
resid = zeros(t-lagdiff,k);

index = 1;
error=zeros(k,1);
for i=1:k
    if errortype(i)==1
        error(i)=0;
    else
        error(i)=1;
    end
end

H=zeros(size(resid));

for i=1:k
    b0 = params(index+archP(i)+garchQ(i)+o(i)+1:index+archP(i)+garchQ(i)+o(i)+arlag(i)+const); % wähle regressor aus
    n = length(data);
    nlag = arlag(i);
    x = [ones(n,1) mlag(data(:,i),nlag)]; %kreire Regressions-Matrix
    x = trimr(x,nlag,0); %korrigiere um AR-lag
    reg = trimr(data(:,i),nlag,0);
    reg = reg - x*b0;
    if nlag < lagdiff
        resid(:,i) = trimr(reg,lagdiff-nlag,0); %korrigiere  um unterschiedliche laglängen
    else
        resid(:,i) = reg;
    end
    
    index = index + archP(i) + garchQ(i) +o (i) + const + arlag(i) + error(i) + 1; %zähle univariate Parameter weiter
end


[t,k]=size(resid);
index=1;
for i=1:k
    univariateparams = params(index:index+archP(i)+o(i)+garchQ(i));
    if garchtype(:,i)==0
        H(:,i) = egarchcore(resid(:,i), univariateparams(1:end-error(i)), stdEstimate(:,i), archP(i), o(i), garchQ(i), m , t);
    else
        %         set leverage term to zero if model contains not leverage but
        %         power-parameter
        if garchtype(i) == 4
            o2(i) = 0;
        elseif garchtype(i) == 6
            %     special case: APGARCH contains leverage and power-parameter
            o2(i) = 1;
        else
            o2(i) = o(i);
        end
        [simulatedata, H(:,i)] = ADCC_univariate_simulate(univariateparams, resid(:,i), archP(i), o2(i), garchQ(i), garchtype(i), errortype(i), stdEstimate(i));
    end
    index = index+archP(i)+garchQ(i)+o(i)+const+arlag(i)+1;
end

stdresid = resid./sqrt(H);

a = params(index:index+dccP-1);
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
Qinitial = Qbar * (eye(k) - sumA^2 - sumB^2);
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
    likelihoods(j)=k*log(2*pi)+sum(log(H(j,:)))+log(det(Rt(:,:,j)))+stdresid(j,:)*(Rt(:,:,j))^(-1)*stdresid(j,:)';
    logL = logL + likelihoods(j);
end;

Qt = Qt(:,:,(m+1:t+m));
Rt = Rt(:,:,(m+1:t+m));
logL = (1/2)*logL;
likelihoods = [(1/2)*likelihoods(m+1:t+m)]';

