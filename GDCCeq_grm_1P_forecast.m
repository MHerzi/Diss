function [Rt, veclRt]=GDCCeq_grm_1P_forecast(theta,data,optimizer)
% % Calculate the time varying correlations based on the GDCC(1,1)
% % specification of Engle, given a parameters vector theta and a matrix of
% % standardized iid residuals data

if iscell(theta) == 1
    theta = theta{1,1}{1,1};
end

[t,k] = size(data);
P=1;
Q=1;
A = diag(theta(1:P*k));
B = diag(theta(P*k+1:(P+Q)*k));

if strcmp(optimizer,'fminunc')==1
    A=A.^2/(1+A.^2+B.^2);
    b=B.^2/(1+A.^2+B.^2);
end

veclRt=zeros(t,k*(k-1)/2);

Qbar=cov(data);

% Next compute Qt
m = max(P,Q);
Qt = zeros(k,k,t+m);
Rt = zeros(k,k,t+m);
Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
Rt(:,:,1:m) = repmat(Qbar,[1 1 m]);
data = [zeros(m,k);data];
for j=(m+1):t+m+1
    Qt(:,:,j) = Qbar - A'*Qbar*A - B'*Qbar*B;
    for i = 1:P
        Qt(:,:,j) = Qt(:,:,j)+A*(data(j-i,:)'*data(j-i,:))*A;
    end
    for i=1:Q
        Qt(:,:,j) = Qt(:,:,j)+B*Qt(:,:,j-i)*B;
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


