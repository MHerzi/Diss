function [out1,t] = frank_inverse(kappa,n,state)
%function out1 = plackett_rnd(kappa,n)
% Generates n (bivariate) random numbers from the Frank copula
% with parameter kappa.
%
% INPUTS:	kappa, a scalar, the parameter of the Plackett copula, must be greater than 0 (correl=0.5 at about kappa=5.5)
%				n , a scalar, the number of draws required
%				state, an integer to use to seed the random number generator
%
% OUTPUTS: out1, an nx2 matrix of random numbers 
%
% Author: Martin Grziska 

if nargin==1
   n = 1;	
end
if nargin<3
   rand('state',sum(1234*clock));	% setting RNG to new seed according to computer clock time.
else
   rand('state',state);
end

U = rand(n,1);
t = rand(n,1);		% interim variable

V=-1/kappa.*log(1+(t.*(exp(-kappa)-1))./(t+(1-t).*exp(-kappa.*U)));
out1 = [U,V];