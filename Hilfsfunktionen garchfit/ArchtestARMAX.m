function [parameters_opt, errors_opt2, LLF_opt, SEregression_opt, stderrors_opt, robustSE_opt, scores_opt, likelihoods_opt, Indexwithoutconst, Indexwithconst, results_lm2, results_lm1]=ArchtestARMAX(daten,maxL,q)

% Bestimmt automatisch die optimale Laglänge des ARMAX-Modells. Es wir das Akaike-Kriterium
% zur Bestummung der Laglänge verwendet.
% daten: Zeitreihe, über die der Armaxfilter gelegt werden soll
% maxL:  maximale Laglänge, mit der die ARMA-order bestimmt werden soll
% q:     maximale Laglänge für die LM-Tests

% j: maximale Länge des MA-Term
% l:maximale Länge des AR-Term
% Erläuterung zu den Variablen:
% h: mit/ohne Konstante

% 1. Lege einen Armaxfilter über die Zeitreihe, um sie zu "demeanen"
% in den Zeilen steht der jeweilige AR-Term, in den Spalten der jeweilige
% MA-Term, und in der dritten Dimension ob mit Constante oder ohne, wobei ohne Konstante zuerst geschätzt wird
% Bsp.: in arma[2,2,1]: AR:2, MA:2, mit Konstante

[t,k]=size(daten);
for i=1:k
    for h=1:2
        for j=1:maxL
            for l=1:maxL
                [LLF{i}.arma(j,l,h), errors{i}.const{h}.arma{j}.ma(:,l)]=armaxfilter_aic(daten(:,i),h-1,j,l);
                AICvec{i}.const{h}.ar{j}.ma(:,l)=log((errors{i}.const{h}.arma{j}.ma(:,l)'*errors{i}.const{h}.arma{j}.ma(:,l))/t)+(2*(l+j))/t;
                AIC{i}.arma(j,l,h)=AICvec{i}.const{h}.ar{j}.ma(:,l);
            end
        end
    end
end

% Bestimmung der optimalen Laglänge
for i=1:k
    for h=1:2
        [ARMAZeile{i}.const{h}.Wert,ARMA{i}.const{h}.ZeilenIndex]=min(AIC{i}.arma(:,:,h));
%         [ARMA{i}.const{h}.Wert,ARMA{i}.const{h}.SpaltenIndex]=min(min(AIC{i}.arma(:,:,h)));
        [ARMA{i}.const{h}.Wert,ARMA{i}.const{h}.SpaltenIndex]=min(ARMAZeile{i}.const{h}.Wert);
        if h==1
            Indexwithoutconst(:,:,i)=[ARMA{i}.const{h}.ZeilenIndex(1,1) ARMA{i}.const{h}.SpaltenIndex];
        else
            Indexwithconst(:,:,i)=[ARMA{i}.const{h}.ZeilenIndex(1,1) ARMA{i}.const{h}.SpaltenIndex];
        end
    end
end



% Schätzung der Parameter ohne Konstante mit optimaler Laglänge
for i=1:size(daten,2)
[parameters_opt{i}.result, errors_opt{i}.result, LLF_opt{i}.result , SEregression_opt{i}.result, stderrors_opt{i}.result, robustSE_opt{i}.result, scores_opt{i}.result, likelihoods_opt{i}.result]=armaxfilter(daten(:,i),0,Indexwithoutconst(1,1,i),Indexwithoutconst(1,2,i));
end

% maxIndex=max(Indexwithoutconst);
% minIndex=min(Indexwithoutconst);
% maxDiff=maxIndex-minIndex;

% Zurechtschneiden der Error-Vektoren - die unterschiedliche Länge ergibt
% sich durch unterschiedliche AR(MA)-Order - auf die Länge des kleinsten
% ARMA-Vektors; die erst(en) Beobachtungen werden dabei vernachlässigt

for i=1:size(errors_opt,2)
    sizeARMAX(:,i)=size(errors_opt{i}.result,1);
end

minIndex=min(sizeARMAX);


for i=1:size(errors_opt,2)
    if size(errors_opt{i}.result,1)>minIndex;
    errors_opt{i}.result=errors_opt{i}.result(size(errors_opt{i}.result,1)-minIndex+1:size(errors_opt{i}.result,1),:);
    end
end

% Engles' LM-Test zur Überprüfung der Residuen
for i=1:k
    results_lm2{i} = lmtest2(errors_opt{i}.result,q);
end

for i=1:k
    for j=1:size(results_lm2{i}.pval,1)
        if results_lm2{i}.pval(j,:)<.05
            fprintf(1,'Das Modell der Serie %d\n', i)
            fprintf(1,'weist GARCH-Effekte auf, bei Lag %d\n', j)
        else
            fprintf(1,'Das Modell der Serie %d\n', i)
            fprintf(1,'weist !!!!KEINE!!!! Garch Effekte auf, bei Lag %d\n', j)
        end
    end
end

for i=1:k
results_lm1{i} = lmtest1(errors_opt{i}.result,q);
end

errors_opt2=zeros(size(errors_opt{1}.result,1),k);
for i=1:k
    errors_opt2(:,i)=errors_opt{i}.result;
end

    
