function tau=Pattoneq_grm_1P_forecast(theta,data,type,opti)
T = size(data,1);
x = data(:,1);
y = data(:,2);
if  strcmp(type,'SJC')==1
    tau = ones(T,2); psi=ones(T,2);
    tau(1,:) = .15*ones(1,2);
    for jj = 2:T+1
        if jj<=10
            psi(:,1) = theta(1,1) + theta(2,1)*mean(abs(x(1:jj-1)-y(1:jj-1))) + theta(3,1)*psi(jj-1,1);
            psi(:,2) = theta(1,2) + theta(2,2)*mean(abs(x(1:jj-1)-y(1:jj-1))) + theta(3,2)*psi(jj-1,2);
        else
            psi(:,1) = theta(1,1) + theta(2,1)*mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))) + theta(3,1)*psi(jj-1,1);
            psi(:,2) = theta(1,2) + theta(2,2)*mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))) + theta(3,2)*psi(jj-1,2);
        end
        tau(jj,:) =.0001+.85./(1+exp(-psi(jj,:)));
    end
elseif strcmp(type,'Clayton')==1
    tau = -999.999*ones(T,1); psi=zeros(T,1);
    tau(1) = copulafit('clayton',[x,y]);
    for jj = 2:T+1
        if jj<=10
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(1:jj-1)-y(1:jj-1))));
        else
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))));
        end
        if strcmp(opti,'fminunc')
            %    Transformation
            tau(jj) =.001+10./(1+exp(-psi(jj)));
        end
    end
elseif strcmp(type,'Gumbel')
    tau = -999.999*ones(T,1); psi=zeros(T,1);
    tau(1) = copulafit('gumbel',[x,y]);
    for jj = 2:T+1
        if jj<=10
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(1:jj-1)-y(1:jj-1))));
        else
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))));
        end
        if strcmp(opti,'fminunc')
            %    Transformation
            tau(jj) = 1.001+10./(1+exp(-psi(jj)));
        end
    end
end