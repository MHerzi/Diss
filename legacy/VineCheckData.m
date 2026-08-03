[data_StudentT] = VineCheckData(U_new);

% Estimates several bivariate Copulas to determine structure for
% first decomposition of Vine. For StudentT:CopulaPairs with lowest d.o.f are
% selected, because they show most dependence; for Gumbel or Clayton,
% Copulas with highest parameeters are selected; because they show most
% dependence

% estimate all possible pairs
[t,k]=size(U_new);

% for StudentT
for j=1:k
    for i=j+1:k
        data_est=[U_new(:,j) U_new(:,i)];
        [RHOHAT,nuhat(j,i)] = copulafit('t',data_est);
    end
end
% ordne die Daten nach der Abhängigkeit, d.h. kleinste d.o.f. spiegelt
% höchste Abhängigkeit wieder

% for both archimedean Copulas: order pairs for max dependence
% for Gumbel
for j=1:k
    for i=j+1:k
        data_est=[U_new(:,j) U_new(:,i)];
        [gumbparam(j,i)] = copulafit('Gumbel',data_est);
    end
end

% for Clayton
% je größer der parameter desto höher abhängigkeit
for j=1:k
    for i=j+1:k
        data_est=[U_new(:,j) U_new(:,i)];
        [clayparam(j,i)] = copulafit('Clayton',data_est);
    end
end

% for rotated Clayton
% je größer der parameter desto höher abhängigkeit
for j=1:k
    for i=j+1:k
        data_est=[1-U_new(:,j) 1-U_new(:,i)];
        [rotclayparam(j,i)] = copulafit('Clayton',data_est);
    end
end

% % fill zeros in matrix, so min value can be >0
% h=1;
% for j=1:k-1
%     for i=1:h
%         nuhat(j,i) = 1000;
%     end
%     h=h+1;
% end
%
% % find pairs with lowest d.o.f. for first row
% % number of pairs in first row:
% for i=1:k-1
% %     find min in row
%     [Holder b_row(i,:)] = min(nuhat);
%     [Holder b_col(i)] =min(min(nuhat));
%     minval(i,:)= [b_row(i,b_col(i)) b_col(i)];
%     nuhat(minval(i,1),minval(i,2)) = 10000;
% end
% % order data according to lowest d.o.f. for first row


