function [u,upareto] = paretocdf(data)
% Funktion zur Bestimmung der c.d.f. mit matlab-paretotials. Die tails
% bestehen aus jeweils 10% der Werte, zum smoothen des "Mittelteils" wird
% ein kernel aeingesetzt
% INPUTS:
%        data : txk array standardisierter Residuen
% OUTPUTS:
%        u: U(0,1)-Variablen
%  upareto: Matlab paretotails Objekt

[t,k] = size(data);
upareto=cell(k,1);
for i=1:k
    upareto{i} = paretotails(data(:,i),.1,.9,'kernel');
%     upareto{i} = paretotails(data(:,i),.1,.9,'ecdf');
    u(:,i) = upareto{i}.cdf(data(:,i));
end