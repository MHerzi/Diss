function [logL, Rt, likelihoods, Qt] = GDCC_likelihood(params, stdresid, P, Q, epsilon)
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
% last modification: 08, 23, 2010

[t,k] = size(stdresid);
A = diag(params(1:(P*k)));       % ARCH for all residuals
B = diag(params((P*k)+1:(P*k)+Q*k));        % GARCH

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
violatedPSD = 0;
Qinitial = Qbar - A'*Qbar*A - B'*Qbar*B;

for j = (m+1):t+m
   Qt(:,:,j) = Qinitial;
   for i=1:P
     Qt(:,:,j) = Qt(:,:,j) + A'*(stdresid(j-i,:)'*stdresid(j-i,:))*A;
   end
   for i = 1:Q
      Qt(:,:,j) = Qt(:,:,j) + B'*Qt(:,:,j-i)*B;
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

% Alternative Spezifikation mit a und b als Vektoren
% a = params(1:(P*k));       % ARCH for all residuals
% b = params((P*k)+1:(P*k)+Q*k);        % GARCH
% 
% % First compute Qbar, the correlation matrix of the standardized residuals,
% % and Nbar, the correlation matrix of the standardized residuals,
% % conditional on the standardized residuals being negative
% Qbar = cov(stdresid);
% 
% % Next compute Qt
% m = max(P,Q);
% Qt = zeros(k,k,t+m);
% Rt = zeros(k,k,t+m);
% Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
% Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
% logL = 0;
% likelihoods = zeros(1,t+m);
% 
% stdresid = [zeros(m,k); stdresid];
% violatedPSD = 0;
% Qinitial = Qbar.*(ones(k,k)-(a*a')-(b*b'));
% 
% for j = (m+1):t+m
%    Qt(:,:,j) = Qinitial;
%    for i=1:P
%      Qt(:,:,j) = Qt(:,:,j) + a*a'.*(stdresid(j-i,:)'*stdresid(j-i,:));
%    end
%    for i = 1:Q
%       Qt(:,:,j) = Qt(:,:,j) + b*b'.*Qt(:,:,j-i);
%    end
%    Rtemp = Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
%    Rtemp = Rtemp - diag(diag(Rtemp)) + eye(k);
%    Rt(:,:,j) = Rtemp;
%    maxmax = max(max(Rt(:,:,j)));
%    minmin = min(min(Rt(:,:,j)));
%    if maxmax > 1 || minmin < -1
%        violatedPSD = 1;
%    end
%    likelihoods(j) = log(det(Rt(:,:,j))) + stdresid(j,:)*(Rt(:,:,j))^(-1)*stdresid(j,:)';
%    logL = logL + likelihoods(j);
% end;
% 
% Qt = Qt(:,:,(m+1:t+m));
% Rt = Rt(:,:,(m+1:t+m));
% logL = (1/2)*logL;
% likelihoods = (1/2)*likelihoods(m+1:t+m);
% 
% if ~isreal(logL) || violatedPSD 
% %    disp('Imag')
% %    params
%    logL=10E+15;
% end
% 
% if isinf(logL)
% %    disp('Inf')
% %    params
%    logL=10E+15;
% end
