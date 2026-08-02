% GED QQ-Plot
empirical = empiricalCDF(data);
empdata = sort(empirical);
for i=1:size(data,1)
U(i,:)=gedinv(empdata(i),1.4);
end

U = sort(U);