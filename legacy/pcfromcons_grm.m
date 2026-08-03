function [mat,veclm]=pcfromcons(cons)
% given the consentration matrix 'cons' (the inverse of the covariance matrix)
% this function calculates the partial correlation matrix 'mat'
if size(cons,3)==1
N=size(cons,1);
mat=zeros(N,N);
for i=1:N
    for j=(i+1):N
        mat(i,j)=-cons(i,j)/sqrt(cons(i,i)*cons(j,j));
        mat(j,i)=mat(i,j);
    end
    mat(i,i)=1;
end
veclm=vecl(mat);
elseif size(cons,3)>1 && size(cons,1)==size(cons,2);
    [N,N,T]=size(cons); veclm=zeros(T,N*(N-1)/2);
    mat=zeros(N,N,T);
    for k=1:T
        for i=1:N
            for j=(i+1):N
            mat(i,j,k)=-cons(i,j,k)/sqrt(cons(i,i,k)*cons(j,j,k));
            mat(j,i,k)=mat(i,j,k);
            end
        mat(i,i,k)=1;
        end
        veclm(k,:)=vecl(mat(:,:,k))';
    end
else
    error('cons is a square matrix or a 3d array of square matrices');
end