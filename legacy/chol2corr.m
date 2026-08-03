% Transformieren der Cholesky Faktoren in Form eines Vektors in eine
% Korrelationsmatrix. Die Cholesky Faktoren repräsentieren die obere
% Cholesky Dreiecksmatrix.
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function [Korrelationsmatrix, CFM, PD_Test] = chol2corr(DatasetSize, CholeskyParam);

b = DatasetSize;
cp = CholeskyParam;

% Prüfen dass die Cholesky Parameter als 1xn Vektor übergeben werden
[s1 s2] = size(cp);
if s1 > s2
    cp = cp';
    [s1 s2] = size(cp);
end
if s1 ~= 1
    error('Die Cholesky Faktoren müssen als 1xn Vektor übergeben werden');
end
    
% Überprüfen ob richtige Anzahl von CholeskyParametern vorhanden
if size(cp, 2) ~= (b^2-b)/2+b
    error('falsche Anzahl von Cholesky Parametern')
end

% CholeskyParam in Matrixform bringen
K = zeros(b);
n = 1;
for i = 1:b
    K(i,i:end) = cp(n:n+b-i);
    n=n+b-i+1;  
end
% Cholesky Kovarianzmatrix bestimmen und Korrelationsmatrix mit corrcov
% auslesen
Korrelationsmatrix = corrcov(K'*K);
% Überprüfen dass die Korrelationsmatrix positive definit ist
% und als Constraint an ceq übergeben. Falls die Korrelationsmatrix nicht
% pos. definit ist wird sie mit Hilfe von RSC_validcorr nachadjustiert.
[CFM, PD_Test] = cholcov(Korrelationsmatrix, 0);
if PD_Test ~= 0
    Korrelationsmatrix = validcorr(Korrelationsmatrix);
end
