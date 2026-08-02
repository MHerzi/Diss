function [logL, Rt, likelihoods, Qt] = ADCC_likelihood_grm(params, stdresid, P, Q, epsilon)
% PURPOSE:
%        Restricted likelihood for use in the ADCC_MVGARCH estimation and
%        returns the likelihood of the 2SQMLE estimates of the ADCC parameters
%
% USAGE:
%        [logL, Rt, likelihoods, Qt] = ADCC_likelihood(params, stdresid, P, Q)
%
% INPUTS:
%    params      - A 2*P+Q by 1 vector of parameters of the form [dccPparameters; dccQparameters, adccPparameters]
%    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
%
% OUTPUTS:
%    logL        - The calculated Quasi-Likelihood (negative)
%    Rt          - a k x k x t 3 dimesnaional array of conditional correlations
%    likelihoods - a t by 1 vector of quasi likelihoods (negative)

%
%
% COMMENTS:
%
%
% Author: Christoph Schleicher
% based on the DCC code by Kevin Sheppard

[t,k] = size(stdresid);
a          = params(1:P);       % ARCH for all residuals
a_negative = params(P+1:2*P);   % ARCH for negative residuals
b = params(2*P+1:2*P+Q);        % GARCH
sumA = eye(k)*sum(a);
sumA_negative = eye(k)*sum(a_negative);
sumB = eye(k)*sum(b);

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
Qinitial = Qbar * (eye(k) - sumA.^2 - sumB.^2) - Nbar_negative * sumA_negative.^2;
for j = (m+1):t+m
    Qt(:,:,j) = Qinitial;
    for i=1:P
        Qt(:,:,j) = Qt(:,:,j) + a(i)*(stdresid(j-i,:)'*stdresid(j-i,:))*a(i);
        Qt(:,:,j) = Qt(:,:,j) + a_negative(i)*(negativeStdresid(j-i,:)'*negativeStdresid(j-i,:))*a_negative(i);
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
    likelihoods(j) = log(det(Rt(:,:,j))) + stdresid(j,:)*(Rt(:,:,j))^(-1)*stdresid(j,:)';
    logL = logL + likelihoods(j);
end;

Qt = Qt(:,:,(m+1:t+m));
Rt = Rt(:,:,(m+1:t+m));
logL = (1/2)*logL;
likelihoods = (1/2)*likelihoods(m+1:t+m);

if ~isreal(logL) || violatedPSD
    logL=10E+15;
end

if isinf(logL)
    logL=10E+15;
end

% Modification by Martin Grziska
if isnan(logL)
    logL = 10E+15;
end

if any(isnan(params)) || ~isreal(params)
    logL=10E+15;
end
