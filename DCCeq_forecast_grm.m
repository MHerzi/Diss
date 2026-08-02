function [Rt, veclRt]=DCCeq_forecast_grm(theta,data,optimizer)

% make 1-period forecast for DCC

if iscell(theta) == 1
    theta = theta{1,1}{1,1};
end

[t,k] = size(data);
P=1;
Q=1;
a = theta(1:P);
b = theta(P+1:P+Q);
if strcmp(optimizer,'fmincon')==1
    a=theta(1); b=theta(2);
elseif strcmp(optimizer,'fminunc')==1
    a=theta(1)^2/(1+theta(1)^2+theta(2)^2);
    b=theta(2)^2/(1+theta(1)^2+theta(2)^2);
end
sumA = sum(a);
sumB = sum(b);
veclRt=zeros(t,k*(k-1)/2);

Qbar=cov(data);

% Next compute Qt
m = max(P,Q);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
% use residual at time t to make forecast for t+1
data = [zeros(m,k);data];
for j=(m+1):t+m+1
    Qt(:,:,j) = Qbar*(1-sumA-sumB);
    for i = 1:P
        Qt(:,:,j) = Qt(:,:,j)+a(i)*(data(j-i,:)'*data(j-i,:));
    end
    for i=1:Q
        Qt(:,:,j) = Qt(:,:,j)+b(i)*Qt(:,:,j-i);
    end
    Rt(:,:,j) = Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
end;

% shift matrix so that at forecast for t+1 is at t
Qt = Qt(:,:,(m+1+1:t+m+1));
Rt = Rt(:,:,(m+1+1:t+m+1));

for j=1:t
veclRt(j,:)=rho2theta(Rt(:,:,j));
end
