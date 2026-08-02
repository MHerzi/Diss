function tau=Pattoneq_mix_grm(theta,datapat,type)
T = size(datapat,1);
x = datapat(:,1);
y = datapat(:,2);

if strcmp(type,'clayton')==1
    psi=zeros(T,1);
    psi(1) = log(copulafit('clayton',datapat)); %startparameter ist unconditional copula; muss noch logarithmiert werden, da unten exponiert wird 
%     psi(1) = theta(1); % Die Konstante des ARMA Prozess ist der transponierte Mean des
    % Copulaparameters
    for jj = 2:T
        if jj<=10
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(1:jj-1)-y(1:jj-1))));
        else
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))));
        end
    end
    tau = exp(psi) + 1e-6;
elseif strcmp(type,'gumbel')
    psi=zeros(T,1);
    psi(1) = log(copulafit('gumbel',datapat));%startparameter ist unconditional copula; muss noch logarithmiert werden, da unten exponiert wird 
%     psi(1)=theta(1); % Die Konstante des ARMA Prozess ist der transponierte Mean des
    % Copulaparameters
    for jj = 2:T
        if jj<=10
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(1:jj-1)-y(1:jj-1))));
        else
            psi(jj) = theta(1)+ theta(2)*psi(jj-1) + theta(3)*(mean(abs(x(jj-10:jj-1)-y(jj-10:jj-1))));
        end
    end
    tau = exp(psi) + 1 + 1e-6;
end
