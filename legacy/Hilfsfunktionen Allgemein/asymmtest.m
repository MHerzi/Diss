function[]=asymmtest(data)
% Test von Returns auf Asymmetrien
% Input: Returnzeitreihen
% Nullhypothese:E[r^2_(it)/r_(it-1)<0]=E[r^2_(it)/r_(it-1)>0]

[t,k]=size(data);

% Returns von Null aus den Daten löschen
% [row1,Holder]=find(x(:,1)==0);
% [row2,Holder]=find(x(:,2)==0);
% [row3,Holder]=find(x(:,3)==0);

% !!!!!!! Achtung muss manuell mehrmals gestartet werden
for i=1:size(data,1)
    for j=1:k
        if data(i,j)==0
            data(i,:)=[];
        end
    end
end


% Indicatorvariable für negative Returns 
[t,k]=size(data);
dataIndicatornegative=zeros(t,k);
for j=1:k
    for i=1:t
        if data(i,j)<0
            dataIndicatornegative(i,j)=1;
        else
            dataIndicatornegative(i,j)=0;
        end
    end
end

% Indicatorvariable für positive Returns
dataIndicatorpositive=zeros(t,k);
for j=1:k
    for i=1:t
        if data(i,j)>0
            dataIndicatorpositive(i,j)=1;
        else
            dataIndicatorpositive(i,j)=0;
        end
    end
end

% Berechnen der (gelaggten) Indikatorvariablen für positive und negative Returns 
lag=1;
lagReturn=mlag(data,lag);
lagReturn=lagReturn(lag+1:end,:);
data=data(lag+1:end,:);
dataIndicatorlagnegative=mlag(dataIndicatornegative,lag);
dataIndicatorlagnegative=dataIndicatorlagnegative(lag+1:end,:);
AsymmReturnnegative=dataIndicatorlagnegative.*(data);
dataIndicatorlagpositive=mlag(dataIndicatorpositive,lag);
dataIndicatorlagpositive=dataIndicatorlagpositive(lag+1:end,:);
AsymmReturnpositive=dataIndicatorlagpositive.*(data);

% Berechne E[r^2/r_(it-1)<0]
% Kalkuliere die Varianz der Returns unter der Bedingung, dass der Return
% der Vorperiode negativ war
% 1.Lösche die Zeilen in AsymmReturnnegative(quadrierte Returns) in denen eine Null vorkommt,
% der Return des Vortages also positiv war

% Aufspalten der Vektoren
for i=1:size(AsymmReturnnegative,2)
    AsymmReturnnegative_neu(i).result=AsymmReturnnegative(:,i);
end

% !!!!!!!!!!!!Muss mehrmals von Hand gestartet werden
for i=1:size(AsymmReturnnegative_neu,2)
    for j=1:size(AsymmReturnnegative_neu(i).result,1)
        if AsymmReturnnegative_neu(i).result(j)==0
            AsymmReturnnegative_neu(i).result(j)=[];
        end
    end
end

% Aufspalten der Vektoren
for i=1:size(AsymmReturnpositive,2)
    AsymmReturnpositive_neu(i).result=AsymmReturnpositive(:,i);
end

% !!!!!!!!!!!!Muss mehrmals von Hand gestartet werden
for i=1:size(AsymmReturnpositive_neu,2)
    for j=1:size(AsymmReturnpositive_neu(i).result,1)
        if AsymmReturnpositive_neu(i).result(j)==0
            AsymmReturnpositive_neu(i).result(j)=[];
        end
    end
end

% Berechnung der Varianz
for i=1:size(AsymmReturnnegative_neu,2)
AssetVarianz_positive(:,i)=(AsymmReturnpositive_neu(i).result'*AsymmReturnpositive_neu(i).result)/(size(AsymmReturnpositive_neu(i).result,1)-1);
AssetVarianz_negative(:,i)=(AsymmReturnnegative_neu(i).result'*AsymmReturnnegative_neu(i).result)/(size(AsymmReturnpositive_neu(i).result,1)-1);
end


[a,b]=size(AsymmReturnpositive);
[c,d]=size(AsymmReturnnegative);
% Berechnen des arithmetischen Mittels der Stabw der Return-Zeitreihen
Volabar=sqrt(AssetVarianz_positive*(b*(a-1))+AssetVarianz_negative*(d*(c-1)))/((a-1)*b+(c-1)*d);


% Berechnen des kritischen Wertes
criticalval=(AssetVarianz_positive-AssetVarianz_negative)/Volabar;

squareReturn=data.^2;
squareReturn=squareReturn(2:end,:);
lagReturn=lagReturn(2:end,:);
Returndivide=squareReturn./lagReturn;
sumReturndivide=sum(Returndivide)/(t-lag);

% Problem beim ablesen des Wertes der t-Verteilung: die beiden Samples der
% positiven und negativen Varianzen haben untersciedliche size!!!!
dof=t-lag-1+t-lag-1;
tval=(tinv(.99,dof)*sqrt(var(Returndivide)))/sqrt(t-lag);
sumIndicatornegative=sum(dataIndicatorlagnegative);
sumIndicatorpositive=sum(dataIndicatorlagpositive);

sumReturnnegative=sum(AsymmReturnnegative);
sumReturnpositive=sum(AsymmReturnpositive);

AsymmetricResult=sumReturnnegative./sumIndicatornegative-sumReturnpositive./sumIndicatorpositive;