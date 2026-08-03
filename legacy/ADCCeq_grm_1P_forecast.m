function [Rt, veclRt]=ADCCeq_grm_1P_forecast(theta,data,optimizer)

% One-period forecast for ADCC
if iscell(theta) == 1
    theta = theta{1,1}{1,1};
end

[t,k] = size(data);
P=1;
Q=1;
a = theta(1:P);
g = theta(1:P+1:2*P);
b = theta(2*P+1:2*P+Q);
if strcmp(optimizer,'fmincon')==1
    a=theta(1); g = theta(3); b=theta(2);
elseif strcmp(optimizer,'fminunc')==1
    a=theta(1)^2/(1+theta(1)^2+theta(2)^2+theta(3)^2);
    g=theta(2)^2/(1+theta(1)^2+theta(2)^2+theta(3)^2);
    b=theta(2)^2/(1+theta(1)^2+theta(2)^2+theta(3)^2);
end
sumA = sum(a);
sumG = sum(g);
sumB = sum(b);
veclRt=zeros(t,k*(k-1)/2);
Qbar=cov(data);

% Next compute Qt
m = max(P,Q);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
data = [zeros(m,k);data];
data_neg = data.*(data<0);
Nbar = cov(data_neg);
for j=(m+1):t+m+1
    Qt(:,:,j) = Qbar*(1-sumA^2-sumB^2)-sumG^2*Nbar;
    for i = 1:P
        Qt(:,:,j) = Qt(:,:,j)+a(i)*(data(j-i,:)'*data(j-i,:))*a(i);
        Qt(:,:,j) = Qt(:,:,j)+g(i)*(data_neg(j-i,:)'*data_neg(j-i,:))*g(i);
    end
    for i=1:Q
        Qt(:,:,j) = Qt(:,:,j)+b(i)*Qt(:,:,j-i)*b(i);
    end
    Rtemp = Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
    Rtemp = Rtemp - diag(diag(Rtemp)) + eye(k);
    Rt(:,:,j) = Rtemp;
end;

Qt = Qt(:,:,(m+1:t+m+1));
Rt = Rt(:,:,(m+1:t+m+1));

for j=1:t+1
    veclRt(j,:)=rho2theta(Rt(:,:,j));
end
