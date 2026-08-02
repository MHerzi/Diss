function [Rt, veclRt]=ADCCeq_grm(theta,data,optimizer)
% % Calculate the time varying correlations based on the DCC(1,1)
% % specification of Engle, given a parameters vector theta and a matrix of
% % standardized iid residuals data

if iscell(theta) == 1
    theta = theta{1,1}{1,1};
end

[t,k] = size(data);
P=1;
Q=1;
a = theta(1:P);
g = theta(P+1:2*P);
b = theta(2*P+1:2*P+Q);
if strcmp(optimizer,'fminunc')==1
    a=a^2/(1+a^2+g^2+g^2);
    g=g^2/(1+a^2+g^2+b^2);
    b=theta(2)^2/(1+a^2+g^2^2+b^2);
end
sumA = sum(a);
sumG = sum(g);
sumB = sum(b);
veclRt=zeros(t,k*(k-1)/2);
Qbar = cov(data);
Nbar = cov(data.*(data<0));

% Next compute Qt
m = max(P,Q);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
data = [zeros(m,k);data];
data_neg = data.*(data<0);
for j=(m+1):t+m
    Qt(:,:,j) = Qbar*(1-sumA^2-sumB^2)-sumG^2*Nbar;
    for i = 1:P
        Qt(:,:,j) = Qt(:,:,j)+a(i)*(data(j-i,:)'*data(j-i,:))*a(i);
        Qt(:,:,j) = Qt(:,:,j)+g(i)*(data_neg(j-i,:)'*data_neg(j-i,:))*g(i);
    end
    for i=1:Q
        Qt(:,:,j) = Qt(:,:,j)+b(i)*Qt(:,:,j-i)*b(i);
    end
    Rt(:,:,j) = Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
end;

Qt = Qt(:,:,(m+1:t+m));
Rt = Rt(:,:,(m+1:t+m));

for j=1:t
    veclRt(j,:)=rho2theta(Rt(:,:,j));
end
