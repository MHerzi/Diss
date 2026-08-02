function [out] = empiricalCopula(data)
% Schätzung der multivariaten empirischen Copula
% Input: Datenmatrix (t x k) von U(0,1) verteilten Variablen
% Output: empirische Copula
% Author: Martin Grziska
% Datum: 23.04.2009
% Die Formel für die empirische (bivariate) Copula stellt sich wie folgt
% dar:
% C_n(u,v) = 1/n * sum_{i=1}^n 1( R_i/(n+1) <= u, S_i/(n+1) <= v ), wobei 
% R_i und S_i die Ränge von u und v sind.
% Praktisch bedeutet dies man für ein beliebiges Wertepaar (z.B. an der 
% Stelle i) in die jeweilige
% Spalte des Wertes schaut und zählt wieviele Werte kleiner sind als der
% Wert an der Stelle i. Das Minimum aus den so ermittelten Zahlen
% pro Zeile wird dann durch n+1 geteilt und man hat den Wert der
% empirischen Copula an der Stelle i.

[t,k] = size(data);

% Finde die Daten mit den kleinsten Wert
[mindata,I] = min(data);
[mindata2,I2] = min(mindata);

% Ordne den Daten eine Spalte von natürlichen Zahlen zu mit den Werten von
% 1:t, um ihnen später die entsprechende Stelle zuordnen zu können
temp1 = [data [1:1:t]'];

% Sortiere die Daten nach der Spalte die den kleinsten Wert enthält
temp1 = sortrows(temp1,I2);

numbpairs1=zeros(t,k);
numbpairs2=zeros(t,1);
for i=1:t
    for j=1:k
%        In dem Feld(i,j) steht die Anzahl der Elemente die kleiner als der
%        Wert im Feld(i,j) der Matrix temp1 ist
        numbpairs1(i,j) = sum(temp1(1:i,j)<=temp1(i,j));
%         Bestimme das Minimum der pro Zeile der oben ermittelten Zahl,
%         dies bestimmt die Anzahl der Werte di
        numbpairs2(i) = min(numbpairs1(i,:));
    end
end

numbpairs3=(numbpairs2)./(t);

% Ordne den Paaren wieder ihre Ausgangsposition zu
temp2=sortrows([numbpairs3 temp1(:,end)] ,2);

% die empirische Copula
out=temp2(:,1);