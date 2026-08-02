function [Rt, veclRt]=TVCeq_grm_forecast_1P(theta,data,optimizer)
% Calculate the time varying correlations based on the TVC(1,1)
% specification of Tse & Tsui, given a parameters vector theta and a matrix
% of standardized iid residuals data

[T,N]=size(data);
Rt=zeros(N,N,T); 
Rtbar=corr(data);
if strcmp(optimizer,'fmincon')==1
    a=theta(1); b=theta(2);
elseif strcmp(optimizer,'fminunc')==1
%     da keine Restriktionen bei der Optimierung eingehalten werden können,
%     müssen die Parameter selbst restringiert werden
    a=theta(1)^2/(1+theta(1)^2+theta(2)^2);
    b=theta(2)^2/(1+theta(1)^2+theta(2)^2);
end
veclRt=zeros(T,N*(N-1)/2);
uu=ceil(1.5*N+1);
% uu muss mindestens 3 sein, da zwei Beobachtungen für die
% Korrelationsmatrix benötigt werden. Da diese aber mit einem lag von ! in
% die Gleichung eingeht werden mind. 3 Beobachtungen benötigt
if uu < 3
    uu = 3;
end
for i=1:T+1
    if i<uu
    Rt(:,:,i)=Rtbar;
    else
    Rt(:,:,i)=Rtbar*(1-a-b)+a*corr(data((i-uu+1):(i-1),:))+b*Rt(:,:,i-1);
    end
%     veclRt(i,:)=vecl(Rt(:,:,i))';
end

% für den Forecast interessiert nur der letzte Wert
veclRt = vecl(Rt(:,:,end));
