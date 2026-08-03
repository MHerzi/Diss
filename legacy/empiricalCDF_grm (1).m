function out1 = empiricalCDF_grm(data)
% Computes empirical cdf
%
%  INPUTS: 	data, a Txk matrix of data (organised in columns)
%
%  OUTPUTS: a Txk matrix of the empirical CDF
%
%  Martin Grziska
%
%  2 October 2008

[T,k] = size(data);
out1 = -999.99*ones(T,k);
for jj = 1:k
    temp = [data(:,jj),(1:1:T)'./(T+1)];
    temp2 = sortrows(temp,1);
    temp3 = sortrows(temp2,2);
    out1(:,jj) = [temp2(:,1) temp3(:,2)];
end
   