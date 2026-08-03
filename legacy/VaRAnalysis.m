function [Output] = VaRAnalysis(VaR);

% exceedence VaR nur für negative returns
PuLneg = VaR.PuL.*(VaR.PuL<0);
% exceedence VaR for den (99/1) siehe Christoffersen (2003)
for i=1:size(VaR.VaRPortfolio991,1)
    if PuLneg(i) > VaR.VaRPortfolio991(i) %wenn die PuL < dem (negativen) VaR ist liegt eine Verletzung vor
        VaRexc991(i) = 0;
    else
        VaRexc991(i) = (VaR.PuL(i)-VaR.VaRPortfolio991(i));
    end
end

% exceedence VaR for den (95/1) siehe Christoffersen (2003)
for i=1:size(VaR.VaRPortfolio951,1)
    if PuLneg(i) > VaR.VaRPortfolio951(i) %wenn die PuL < dem (negativen) VaR ist liegt eine Verletzung vor
        VaRexc951(i) = 0;
    else
        VaRexc951(i) = (VaR.PuL(i)-VaR.VaRPortfolio951(i));
    end
end

% exceedence VaR for den (90/1) siehe Christoffersen (2003)
for i=1:size(VaR.VaRPortfolio901,1)
    if PuLneg(i) > VaR.VaRPortfolio901(i) %wenn die PuL < dem (negativen) VaR ist liegt eine Verletzung vor
        VaRexc901(i) = 0;
    else
        VaRexc901(i) = (VaR.PuL(i)-VaR.VaRPortfolio901(i));
    end
end

