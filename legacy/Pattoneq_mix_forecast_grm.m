function tau = Pattoneq_mix_forecast_grm(theta,datapat,type)
T = size(datapat,1);
x = datapat(:,1);
y = datapat(:,2);

if strcmp(type,'clayton')==1
    tau = -999.999*ones(T,1); psi=zeros(T,1);
    tau(1) = copulafit('clayton',[x,y]);
    psi(1) = tau(1);
    for jj = 2:T+1
        if jj<=10
            psi(jj) = theta(1)+ theta(2)*tau(jj-1) + theta(3)*(mean(abs(x(1:jj-1)-y(1:jj-1))));
        else
            psi(jj) = theta(1)+ theta(2)*tau(jj-1) + theta(3)*(mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))));
        end
        tau(jj,:) =1./(1+exp(-psi(jj,:)));
    end
elseif strcmp(type,'gumbel')
    tau = -999.999*ones(T,1);
    psi=zeros(T,1);
    tau(1) = copulafit('gumbel',[x,y]);
    psi(1)=tau(1);
    for jj = 2:T+1
        if jj<=10
            psi(jj) = theta(1)+ theta(2)*tau(jj-1) + theta(3)*(mean(abs(x(1:jj-1)-y(1:jj-1))));
        else
            psi(jj) = theta(1)+ theta(2)*tau(jj-1) + theta(3)*(mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))));
        end
        tau(jj,:) =1./(1+exp(-psi(jj,:)));
    end
end

% shift matrix so that at t forecast from t to t+1
tau = tau(2:end);