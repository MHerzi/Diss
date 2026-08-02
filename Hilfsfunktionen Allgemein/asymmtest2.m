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
% for i=1:size(data,1)
%     for j=1:k
%         if data(i,j)==0
%             data(i,:)=[];
%         end
%     end
% end


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
% Löschen der Zeilen mit einer Null, also der Zeilen, die Returns
% enthalten, die einem positiven Schock in der Vorperiode folgen.
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
% Löschen der Zeilen mit einer Null, also der Zeilen, die Returns
% enthalten, die einem negativen Schock in der Vorperiode folgen.
for i=1:size(AsymmReturnpositive_neu,2)
    for j=1:size(AsymmReturnpositive_neu(i).result,1)
        if AsymmReturnpositive_neu(i).result(j)==0
            AsymmReturnpositive_neu(i).result(j)=[];
        end
    end
end

% Berechnung der (Sample-)Varianz
for i=1:size(AsymmReturnnegative_neu,2)
AssetVarianz_positive(:,i)=(AsymmReturnpositive_neu(i).result'*AsymmReturnpositive_neu(i).result)/(size(AsymmReturnpositive_neu(i).result,1)-1);
AssetVarianz_negative(:,i)=(AsymmReturnnegative_neu(i).result'*AsymmReturnnegative_neu(i).result)/(size(AsymmReturnpositive_neu(i).result,1)-1);
end


[n1,holder]=size(AsymmReturnpositive);
[n2,holder]=size(AsymmReturnnegative);
% Berechnen des arithmetischen Mittels der Stabw der Return-Zeitreihen
Volabar=sqrt(((AssetVarianz_positive*(n2-1)+AssetVarianz_negative*(n1-1))/((n1-1)+(n2-1)))*(1/(n1-1)+1/(n2-1)));

% Berechnen des kritischen Wertes
criticalval=(AssetVarianz_negative-AssetVarianz_positive)./Volabar;


% beim arithmetischen Mittel entsprechen die Freiheitsgrade: n-1
for i=1:k
    dofpos(:,i)=size(AsymmReturnpositive_neu(i).result,1);
    dofneg(:,i)=size(AsymmReturnnegative_neu(i).result,1);
end

dofges=dofpos+dofneg-2;

for i=1:k
tval(:,i)=(tinv(.99,dofges(:,i)));
end

