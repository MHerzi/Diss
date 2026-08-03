[data_StudentT] = VineStudentTdata(data);

% Estimates several bivariate StudentT-Copulas to determine structure for
% first decomposition of StudentT. CopulaPairs with lowest d.o.f are
% selected

% estimate all possible pairs
[t,k]=size(data);

for j=1:k
    for i=j+1:k
        data_est=[data(:,j) data(:,i)];
        [RHOHAT,nuhat(j,i)] = copulafit('t',data_est);
    end
end

% fill zeros in matrix, so min value can be >0
h=1;
for j=1:k-1
    for i=1:h
        nuhat(j,i) = 1000;
    end
    h=h+1;
end

% find pairs with lowest d.o.f. for first row
% number of pairs in first row:
for i=1:k-1
%     find min in row
    [Holder b_row(i,:)] = min(nuhat);
    [Holder b_col(i)] =min(min(nuhat));
    minval(i,:)= [b_row(i,b_col(i)) b_col(i)];
    nuhat(minval(i,1),minval(i,2)) = 10000;
end


% order data according to lowest d.o.f. for first row
