function [out] = cdfinv_grm(dist,x,nu)
% Mehrere Inverse kumulative Verteilungsfunktionen
% Input: [0,1] Unif verteilte Zeitreihe - x
%        String der Verteilung auf die die Quantilfunktion angewandt werden
%        soll: 'Gauss', 'StudentT,'GED'
%        eventuell zu der Verteilung gehörender Freiheitsgrad - nu

[t,k]=size(x);

out = zeros(t,1);
for i=1:t
    if strcmp(dist,'GED')
        out(i,1) = gedinv(x(i),nu);
    elseif strcmp(dist,'Gauss')
        out(i,1) = norminv(x(i),0,1);
    elseif strcmp(dist,'StudentT')
        out(i,1) = tinv(x(i),nu);
    end
end