true_PuL=zeros(750,1);
for i=1:k
    true_PuL = true_PuL + 1/k*exp(daten(end-750+1:end,i)); %bei Verwendung von Originaldaten muss für die einperiodige Vorhersage
end
true_PuL = log(true_PuL);
VaRviol991=zeros(750,1);
for i =  1:750
    if VaR.VaRPortfolio991(i) > true_PuL(i)
        VaRviol991(i) =  1;
    else VaRviol991(i) =  0;
    end
end
T1 = VaR.SumVaRviol991;
T = 750;
mean_hit = T1/T;
p = 0.01; %Konfidenzniveau
var_It = var(VaRviol991);
MT991 = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit991 = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
% pval = 1-normcdf(abs(MT),0,1);
pval991 = 2*min(normcdf(MT991),1-normcdf(MT991)); %p-Wert für zweiseitigen Test
UncondCoverage991_MT = [MT991 norm_krit991 pval991];
P.pval991=pval991;

VaRviol951=zeros(750,1);
for i =  1:750
    if VaR.VaRPortfolio951(i) > true_PuL(i)
        VaRviol951(i) =  1;
    else VaRviol951(i) =  0;
    end
end
T1 = VaR.SumVaRviol951;
T = 750;
mean_hit = T1/T;
p = 0.05; %Konfidenzniveau
var_It = var(VaRviol951);
MT951 = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit951 = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
% pval = 1-normcdf(abs(MT),0,1);
pval951 = 2*min(normcdf(MT951),1-normcdf(MT951)); %p-Wert für zweiseitigen Test
UncondCoverage951_MT = [MT951 norm_krit951 pval951];
P.pval951=pval951;

VaRviol901=zeros(750,1);
for i =  1:750
    if VaR.VaRPortfolio901(i) > true_PuL(i)
        VaRviol901(i) =  1;
    else VaRviol901(i) =  0;
    end
end
T1 = VaR.SumVaRviol901;
T = 750;
mean_hit = T1/T;
p = 0.10; %Konfidenzniveau
var_It = var(VaRviol901);
MT901 = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit901 = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
% pval = 1-normcdf(abs(MT),0,1);
pval901 = 2*min(normcdf(MT901),1-normcdf(MT901)); %p-Wert für zweiseitigen Test
UncondCoverage901_MT = [MT901 norm_krit901 pval901];
P.pval901=pval901;