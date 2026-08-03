function [U12] = bivGumbel_rnd(theta,T)

% Simuliert nach dem Conditonal Copula Algorithmus von Cherubini S.184
% Daten einer bivariaten Gumbel-Copula
% Input: 
%       theta: Parameter einer bivariaten Gumbel-Copula
%       T    : Skalar (Länge der zu simulierenden Daten)
% Output: 
%       simulierte Daten U1 und U2
% Autor: Martin Grziska, 1. September 2009


% Generiere i.i.d. U(0,1) verteilte Variablen
V = random('unif',0,1,[T 2]);

