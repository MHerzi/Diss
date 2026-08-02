function d = s_white_derivative(f,scores,hess)

% Helfer-Funktion zur Berechnung der ersten Ableitung von s für die
% White(1982) asymptotische Kovarianzmatrix

out = scores'*scores;
sum = out + hess;
% Bilde den vech-operator nach
for i = 1:size(sum,2)
    for j = i:size(sum,2)
        vech(i) = sum(i,j);
    end
end
vech = vech';

fx = feval(f,out,hess);