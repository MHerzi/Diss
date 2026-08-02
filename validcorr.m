function C=validcorr(A)
%Basierend auf Nicholas J. Higham, 2002, 
%"Computing the Nearest Correlation Matrix - A Problem from Finance"
%http://eprints.ma.man.ac.uk/232/01/covered/MIMS_ep2006_70.pdf
S=zeros(size(A));
Y=A;
for k=1:length(A)*1000
    R=Y-S;
    X=Ps(R);
    S=X-R;
    Y=Pu(X);
end
    function X=Ps(A)
        [Q D]=eig(A);
        X=Q*max(D,0)*Q';
    end

    function Y=Pu(X)
        Y=X-diag(diag(X))+eye(length(X));
    end
C=Pu(X);
end