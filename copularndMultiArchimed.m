function U = copularndMultiArchimed(copula,theta,N,D);

% Funktion simuliert unif(0,1)-Verteilte Variablen für multivariate
% archimedische Copulas. Algorithmus nach McNeil, Frey, Embrechts (2008),
% "Quantitative Risk Management", S.223-224
%
% USAGE:
%  U = copularndMultiArchimed(copula,theta,N,D);
%
% INPUTS:
%  copula: string ('clayton','gumbel','rotclayton')
%   theta: dependece Parameter der copula
%       N: Anzahl der zu simulierenden Variablen
%       D: Dimension der zu simulierenden Variablen
% 
% OUTPUT:
%       U: multivariate unif(0,1) simuliert aus archimedischer Copula
%
% Author: Martin Grziska
% Datum: 07/14/2010

copula=lower(copula);
X = random('unif',0,1,N,D);
if strcmp(copula,'clayton') || strcmp(copula,'rotclayton')
    V = randg((1/theta),N,1);
    u = -log(X)./repmat(V,1,D);
    U = (1+u).^(-1/theta);
elseif strcmp(copula,'gumbel')
    V = alphastable((1/theta),1,N,1);
    V = V.*(cos(pi/(2*theta)))^theta;
    u = -log(X)./repmat(V,1,D);
    U = exp(-u.^(1/theta));    
end

% stelle sicher, dass min(U>0) und max(U<1)
U = U + ((U==0)*1e-10);
U = U - ((U==1)*1e-10);

% drehe die Daten für die rotated Clayton
if strcmp(copula,'rotclayton')
    U = 1-U;
end

function [phi] = alphastable(alpha,beta,N,dim)
% generiert alpha-stabilen Funktionswert, siehe "Quantitative Risk
% Management, S. 224 und S.498.
t0  = atan(beta*tan((pi*alpha)/2))/alpha;
Theta = pi*(random('unif',0,1,N,dim) - 0.5);
W = -log(random('unif',0,1,N,dim));
term1 = sin(alpha.*(t0+Theta))./(cos(alpha.*t0).*cos(Theta)).^(1/alpha);
term2 = ((cos(alpha.*t0+(alpha-1).* Theta))./W).^((1 - alpha)/alpha);
phi = term1.*term2;

