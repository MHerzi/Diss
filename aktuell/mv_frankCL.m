% The negative copula log-likelihood of a 
% member of Frank's family
% Taken from Joe (1997), p141.
%
% Martin Grziska
% 30.04.2009

% INPUTS: theta ;
%				data = [U_1 U_2 U_3];

function CL = mv_frankCL(theta,data)
u=data(:,1);
v=data(:,2);
w=data(:,3);

% Dichte der trivariaten Frank-Copula
% CL = theta^2./(exp(theta*u).*exp(theta*v).*exp(theta*z).*(1./exp(theta) - 1)^2.*(((1./exp(theta*u) - 1).*(1./exp(theta*v) - 1).*(1./exp(theta*z) - 1))./(1./exp(theta) - 1)^2 + 1));
% CL = CL - (3*theta^2.*(1./exp(theta*u) - 1).*(1./exp(theta*v) - 1).*(1./exp(theta*z) - 1))./(exp(theta*u).*exp(theta*v).*exp(theta*z).*(1./exp(theta) - 1)^4.*(((1./exp(theta*u) - 1).*(1./exp(theta*v) - 1).*(1./exp(theta*z) - 1))./(1./exp(theta) - 1)^2 + 1).^2); 
% CL = CL + (2*theta^2*(1./exp(theta*u) - 1).^2.*(1./exp(theta*v) - 1).^2.*(1./exp(theta*z) - 1).^2)./(exp(theta*u).*exp(theta*v).*exp(theta*z).*(1./exp(theta) - 1).^6.*(((1./exp(theta*u) - 1).*(1./exp(theta*v) - 1).*(1./exp(theta*z) - 1))./(1./exp(theta) - 1).^2 + 1).^3);
% CL = sum(CL);
% CL = -CL;
CL = (theta^2.*exp(2*theta + 2))./(exp(theta*u).*exp(theta.*v).*exp(theta.*w));
CL =sum(CL);
CL =-CL;
