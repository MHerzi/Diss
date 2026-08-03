function [parameters_opt, errors_opt, LLF_opt, SEregression_opt, stderrors_opt, robustSE_opt, scores_opt, likelihoods_opt, Indexwithoutconst, Indexwithconst,mu, H]=armax_aic_box(daten,maxL)

% Bestimmt automatisch die optimale Laglänge des ARMAX-Modells. Es wir das Akaike-Kriterium
% zur Bestummung der Laglänge verwendet. Danach wird das ARMA-Modell mit
% der optimalen Laglänge geschätzt
% daten: Zeitreihe, über die der Armaxfilter gelegt werden soll
% maxL:  maximale Laglänge, mit der die ARMA-order bestimmt werden soll
% q:     maximale Laglänge für die LM-Tests
% 
% j: maximale Länge des MA-Term
% l:maximale Länge des AR-Term
% Erläuterung zu den Variablen:
% h: mit/ohne Konstante
% 
% Output:
% parameter_opt: Parameter mit der optimalen Laglänge geschätzt
% H:             Test auf white-noise Residuen: H0= Residuen sind white
%                noise (H=0), wenn Residuen keine wihite noise H=1

% 1. Lege einen Armaxfilter über die Zeitreihe, um sie zu "demeanen"
% in den Zeilen steht der jeweilige AR-Term, in den Spalten der jeweilige
% MA-Term, und in der dritten Dimension ob mit Constante oder ohne, wobei
% ohne Konstante zuerst geschätzt wird
% !!!!!!!! [1,1,1] steht für AR(1) MA(0) + Konstante!!!!!!!!!!!!!!!!

[t,k]=size(daten);
for i=1:k
    for h=1:2
        for j=1:maxL
            for l=1:maxL+1
                [LLF{i}.arma(j,l,h), errors{i}.const{h}.arma{j}.ma(:,l), AIC{i}.arma(j,l,h)]=armaxfilter_aic(daten(:,i),h-1,j,l-1);
%                 AICvec{i}.const{h}.ar{j}.ma(:,l)=log((errors{i}.const{h}.arma{j}.ma(:,l)'*errors{i}.const{h}.arma{j}.ma(:,l))/t)+(2*(l+j))/t;
%                 AIC{i}.arma(j,l,h)=AICvec{i}.const{h}.ar{j}.ma(:,l);
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
            Indexwithoutconst(:,:,i)=[ARMA{i}.const{h}.ZeilenIndex(ARMA{i}.const{h}.SpaltenIndex) ARMA{i}.const{h}.SpaltenIndex];
        elseif h==2
            Indexwithconst(:,:,i)=[ARMA{i}.const{h}.ZeilenIndex(ARMA{i}.const{h}.SpaltenIndex) ARMA{i}.const{h}.SpaltenIndex];
        end
    end
end

for i=1:k
Indexwithoutconst(1,2,i)=Indexwithoutconst(1,2,i)-1;
Indexwithconst(1,2,i)=Indexwithconst(1,2,i)-1;
end

% Schätzung der Parameter mit Konstante mit optimaler Laglänge
for i=1:size(daten,2)
[parameters_opt{i}.result, errors_opt{i}.result, LLF_opt{i}.result , SEregression_opt{i}.result, stderrors_opt{i}.result, robustSE_opt{i}.result, scores_opt{i}.result, likelihoods_opt{i}.result]=armaxfilter(daten(:,i),1,Indexwithconst(1,1,i),Indexwithconst(1,2,i));
end

% Der geschätzte Mittelwert
for i=1:k
mu(i)=parameters_opt{i}.result(1)/(1-sum(parameters_opt{i}.result(2:Indexwithconst(1,1,i)+1)));
end

% Check auf Stationarität
for i=1:k
    if abs(sum(parameters_opt{i}.result(2:Indexwithconst(1,1,i)+1))) >=1
        error('nicht stationär Zeitreihe: %d\n',i)
    end
end

% Test ob die Residuen White Noise sind
% Konfidenzniveau: alpha=0.05
for i=1:k
[H(:,i), pValue, Qstat, CriticalValue] = lbqtest(errors_opt{i}.result);
end

for i=1:k
    if H(:,i)==1
        i
        error('Residuen sind nicht white noise')
    end
end