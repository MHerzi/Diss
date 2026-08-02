function [logL, Rt, likelihoods, Qt] = AGDCC_likelihood_grm(params, stdresid, P, Q, epsilon)
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
%    logL        - The calculated Quasi-Likelihood
%    Rt          - a k x k x t 3 dimesnaional array of conditional correlations
%    likelihoods - a t by 1 vector of quasi likelihoods

% 
% 
% COMMENTS:
% 
% 
% Author: Christoph Schleicher, Martin Grziska
% based on the DCC code by Kevin Sheppard

[t,k] = size(stdresid);
a          = diag(params(1:(P*k)));       % ARCH for all residuals
a_negative = diag(params((P*k)+1:2*(P*k)));   % ARCH for negative residuals
b = diag(params(2*(P*k)+1:2*(P*k)+Q*k));        % GARCH

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
Qinitial = Qbar-a'*Qbar-b'*Qbar-a_negative'*Nbar_negative;

for j = (m+1):t+m
   Qt(:,:,j) = Qinitial;
   for i=1:P
     Qt(:,:,j) = Qt(:,:,j) + a'*(stdresid(j-i,:)'*stdresid(j-i,:));
     Qt(:,:,j) = Qt(:,:,j) + a_negative'*(negativeStdresid(j-i,:)'*negativeStdresid(j-i,:));
   end
   for i = 1:Q
      Qt(:,:,j) = Qt(:,:,j) + b'*Qt(:,:,j-i);
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

if ~isreal(logL) || violatedPSD 
%    disp('Imag')
%    params
   logL=10E+15;
end

if ~isreal(params)
%    disp('Imag')
%    params
   logL=10E+15;
end

if isinf(logL)
%    disp('Inf')
%    params
   logL=10E+15;
end
