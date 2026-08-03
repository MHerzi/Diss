function [U12] = bivClayton_rnd(theta,T)

% Simuliert nach dem Conditonal Copula Algorithmus von Cherubini S.184
% Daten einer bivariaten Clayton-Copula
% Input: 
%       theta: Parameter einer bivariaten Gumbel-Copula
%       T    : Skalar (Länge der zu simulierenden Daten)
% Output: 
%       simulierte Daten U1 und U2
% Autor: Martin Grziska, 1. September 2009


% Generiere i.i.d. U(0,1) verteilte Variablen
V = random('unif',0,1,[T 2]);

U1 = V(:,1);

U2 = ((V(:,1).^(-theta)).*(V(:,2).^(-theta/(theta+1))-1)+1).^(-1/theta);

U12 = [U1 U2];
