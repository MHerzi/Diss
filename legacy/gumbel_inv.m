
function [u2] = gumbel_inv(u1,w,theta)

% "numerische" Inverse der Gumbel h-Funktion: gesucht ist die Variable v21,
% die v21 = h^(-1)(v11,w2,theta) erfüllt. Dazu bilde h(v11,v21,theta)-w2 =
% 0 und such v21, dass die Gleichung erfüllt

% kreiere ersten random Vektor
u2 = random('unif',0,1,T,1);
u22=zeros(T,1);
h=zeros(T,1);
% setzte Toleranz für die Differenz
epsilon = 10^(-3);
% wiederhole prozedur sooft bis alles gefüllt ist
while any(u22==0)
    for i=1:T
        for j=1:T
            if u22(i)==0
                h1 = exp(-((-log(u1(i))).^theta + (-log(u2(j))).^theta).^(1/theta)).* 1./u2(j) .* (-log(u2(j))).^(theta-1);
                h2 = ((-log(u1(i))).^theta+(-log(u2(j))).^theta).^(1/theta-1);
                h(j) = h1.*h2;
                if (w(i)-h(j) < epsilon) && (w(i)-h(j)>0)
                    u22(i) = u2(j);
                end
            end
        end
        u2=random('unif',0,1,T,1);
    end
end


