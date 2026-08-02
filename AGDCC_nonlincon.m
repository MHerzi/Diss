function [c, ceq] = AGDCC_nonlincon(params, stdresid, P, Q, epsilon)

% PURPOSE:
%        Nonlinear constraint function for use in the AGDCC_MVGARCH estimation 
%        Qinitial has to be positive-definite to ensure thate Rt is
%        positive definite
% 
% USAGE:
%        [c, ceq] = AGDCC_nonlincon(params, stdresid, P, Q, epsilon)
% 
% INPUTS:
%    params      - A (2*P+Q)*k by 1 vector of parameters of the form [dccPparameters; adccPparameters, dccQparameters]
%    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
%    P           - The innovation order of the ADCC Garch process
%    Q           - The AR order of the ADCC estimator
%    epsilon     - 
% 
% OUTPUTS:
%    c              - The inequality constraint
%    ceq            - The equality constraint

% 
% COMMENTS:
% Stationarity for the AGDCC model is fulfilled if the matrix Qinitial is
% positive definite, see Urga (2007),  A multivariate model for the
% non-normal behavior of Financial Assets;
% http://www.cass.city.ac.uk/conferences/measuringdependence/Files/presenta
% tions/urga.pdf
% 
% 
% Author: Martin Grziska


[t,k] = size(stdresid);

a          = diag(params(1:(P*k)));       
a_negative = diag(params((P*k)+1:2*(P*k)));  
b = diag(params(2*(P*k)+1:2*(P*k)+Q*k));      


Qbar = cov(stdresid);
Nbar_negative = cov(stdresid.*(stdresid < 0));

% Der kleinste Eigenwert muss > 0 sein, dann ist die Matrix positive
% definite (siehe z.B. Cajigas, J.-P., Urga,G.(2007). Dynamic Conditional Correlation
% Models with Asymmetric Multivariate Laplace Innovations.

Qinitial = Qbar - a'*Qbar*a - b'*Qbar*b - a_negative'*Nbar_negative*a_negative;
x_1 = eig(Qinitial);
x_1 = x_1 / abs(max(x_1));
c = min(x_1)*(-1) + epsilon;
ceq = 0;

% % % Es wird die Formel a+b+delta*a_negative<1 von Cappiello et al verwendet;
% % % dazu werden a,b und a_negative von Diagonalmatrizen in Vektoren
% % % umgewandelt und für jedes Element des Vektor wird die Formel angewandt
% delta = max(eig(Qbar^(-1/2)*Nbar_negative*Qbar^(-1/2)));
% a = diag(a);
% b = diag(b);
% a_negative = diag(a_negative);
% for i=1:k
%  c(i) = max(eig(a(i) + b(i) + delta*a_negative(i))) - (1-epsilon);
% end
% 
% ceq=[];
