% Funktion zum Prüfen der Parameter der dynamischen Mixture Copula
%
% INPUT:
% family: min. 1, max. 2 Copulas
% data: [0 1] verteilte Daten
%
% OUTPUT:
% 
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function [nc, family, s1, s2] = copulamix_tv_parameter_check(family, data);

nc = numel(str2double(family)); % Anzahl der Copulas auslesen
family = lower(family);
[s1 s2] = size(data);

% Prüfen dass Familien Struktur als Cell-Struktur übergeben wird
if ~iscell(family)
    family = {family};
end

% Prüfen dass max 2 Copulas benutzt werden. Sonst ist der Rechenaufwand zu
% gewaltig
if nc<1 || nc>2
    error('Es können max. 2 Copulas und min. 1 Copula in die Mixture Struktur implementiert werden');
end

% Prüfen dass Daten als ZeitxIndices eingeben werden
if s2>s1
    data = data';
    [s1 s2] = size(data);
end

% Prüfen dass Daten [0 1] verteilt sind
if max(max(data))>1 || min(min(data))<0
    error('Daten müssen [0 1] verteilt sein');
end

% Prüfen dass max. 10 Indices verwendet werden
if s2<2 || s2>10
    error('Es können minimal 2 und maximal 10 Zeitreihen in den dynamischen Copulas verwendet werden');
end

% Prüfen dass nur erlaubte Copulafamilien verwendet werden
for i = 1:nc
    if sum(strcmp(family{i}, {'clayton' 'gumbel' 't' 'gaussian'})) ~= 1
        error('angegebene Copulafunktion kann nicht benutzt werden');
    end
end

% Prüfen dass nicht mehr als 1 elliptische Copula implementiert werden
tc = 0;
for i = 1:nc
    tc = tc + sum(strcmp(family{i}, {'t' 'gaussian'})); % Test wieviele elliptische Copulas implementiert wurden
end
if tc > 1
    error('Nur 1 ellitpische Copula in der Mixture Struktur macht Sinn!');
end

% Prüfen dass elliptische Copulas immer an erster Stelle der Mixture
% Struktur stehen
for i = 1:nc
    if i >1 && sum(strcmp(family{i}, {'t' 'gaussian'})) 
        family = circshift(family, [0 -(i-1)]);
    end
end














