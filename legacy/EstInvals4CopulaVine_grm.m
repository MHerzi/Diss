function [theta, logL, uncparams]=EstInvals4CopulaVine(data,CopulaSpec)
%%%-----------Estimation of copula vine parameter in many steps-----------
% This function estimates the parameters of a Copula vine by using a
% multistep procedure. Each bivariate copula in the cascade is estimated
% sequentially
% INPUTS:
% data:             A TxN matrix of U(0,1) 
% CopulaSpec:       Structure array with optimization parameters. To obtain
%                   first run the setCopulaVineLLinputs.m
% OUTPUT
% logL:             The log likelihood at the optimum
% theta:            The estimated parameters in vector form. 
% uncparams:        The unconstrained parameters
% ------------------------------------------------------------------------
% Author: Manthos Vogiatzoglou, UoM, 2008 - 2009
% contact at: vogia@yahoo.com
% -------------------------------------------------------------------------
display('**************************************************************')
display('the primary use of this function is to calculate the initial values for the fitCopulaVine.m')
display('---------------------------------------------------------------------------------')
pause(2)
display('however the fitCopulaVine.m is very slow if t - copulas are assumed')
display('and the values obtained from this and fitCopulaVine.m are usually very very close')
display('----------------------------------------------------------------------')
pause(2)
display('therefore if you favor speed over accuracy, you may use this output')
display('----------------------------------------------------------------------')
% error checking and array initialization
if min(min(data))<0 || max(max(data))>1
    data=empiricalCDF(data);
    display('your data is transformed to uniform with the empiricalCDF function')
end
tic
N=size(data,2);
lik=zeros(N-1,N-1);
u=cell(N,N);
theta=zeros(N-1,N-1);
% -------------- the canonical vine decomposition ------------------------
if strcmp(CopulaSpec.decomposition,'CanonicalVine')==1 && strcmp(CopulaSpec.type,'SJC')==0

for j=1:N-1
   fprintf(1,'computing copula parameters for level %d\n',j)
   for i=1:N-j
      if j==1
      [theta(j,i),lik(j,i)]=fitCopula([data(:,1),data(:,i+1)],CopulaSpec);
      else
      [theta(j,i),lik(j,i)]=fitCopula([u{j-1,1}, u{j-1,i+1}],CopulaSpec);
      end
   end
   
   if j<N-1
       for i=1:N-j
       if j==1
       u{j,i}=hfunc(data(:,i+1),data(:,1),theta(j,i),CopulaSpec);    
       else    
       u{j,i}=hfunc(u{j-1,i+1},u{j-1,1},theta(j,i),CopulaSpec);
       end
       end
   end
end
theta=icrIPmat(theta);
% ---------------- canonical vine - SJC copula ----------------------------
elseif strcmp(CopulaSpec.decomposition,'CanonicalVine')==1 && strcmp(CopulaSpec.type,'SJC')==1
theta=cell(N-1,N-1);

for j=1:N-1
   fprintf(1,'computing copula parameters for level %d\n',j)
   for i=1:N-j
      if j==1
      [theta{j,i},lik(j,i)]=fitCopula([data(:,1),data(:,i+1)],CopulaSpec);
      else
      [theta{j,i},lik(j,i)]=fitCopula([u{j-1,1},u{j-1,i+1}],CopulaSpec);
      end
   end
   
   if j<N-1
       for i=1:N-j
       if j==1
       u{j,i}=hfunc(data(:,i+1),data(:,1),theta{j,i},CopulaSpec);    
       else    
       u{j,i}=hfunc(u{j-1,i+1},u{j-1,1},theta{j,i},CopulaSpec);
       end
       end
   end
end
theta=icrIPmat(theta);
% -----------------------------------------------------------------------
% the multi - step code for the D - Vine structure
% -----------------------------------------------------------------------
elseif strcmp(CopulaSpec.decomposition,'DVine')==1 && strcmp(CopulaSpec.type,'SJC')==0
    display('computing copula parameters for level 1');
for i=1:N-1
    [theta(1,i),lik(1,i)]=fitCopula([data(:,i),data(:,i+1)],CopulaSpec);
end
u{1,1}=hfunc(data(:,1),data(:,2),theta(1,1),CopulaSpec);
for k=1:N-3
    u{1,2*k}=hfunc(data(:,k+2),data(:,k+1),theta(1,k+1),CopulaSpec);
    u{1,2*k+1}=hfunc(data(:,k+1),data(:,k+2),theta(1,k+1),CopulaSpec);
end
u{1,2*N-4}=hfunc(data(:,N),data(:,N-1),theta(1,N-1),CopulaSpec);
for j=2:(N-1)
    fprintf(1,'computing copula parameters for level %d\n',j)
    for i=1:(N-j)
        [theta(j,i),lik(j,i)]=fitCopula([u{j-1,2*i-1} u{j-1,2*i}],CopulaSpec);
    end
    if j<N-1
    u{j,1}=hfunc(u{j-1,1},u{j-1,2},theta(j,1),CopulaSpec); 
    if N>4
        for i=1:(N-j-2)
        u{j,2*i}=hfunc(u{j-1,2*i+2},u{j-1,2*i+1},theta(j,i+1),CopulaSpec);
        u{j,2*i+1}=hfunc(u{j-1,2*i+1},u{j-1,2*i+2},theta(j,i+1),CopulaSpec);
        end
    end
    u{j,2*N-2*j-2}=hfunc(u{j-1,2*N-2*j},u{j-1,2*N-2*j-1},theta(j,N-j),CopulaSpec);
    end
end
theta=icrIPmat(theta);
% -------------------- D Vine SJC decomposition ---------------------------
elseif strcmp(CopulaSpec.decomposition,'DVine')==1 && strcmp(CopulaSpec.type,'SJC')==1
display('computing copula parameters for level 1');
theta=cell(N-1,N-1);

for i=1:N-1
    [theta{1,i},lik(1,i)]=fitCopula([data(:,i),data(:,i+1)],CopulaSpec);
end
u{1,1}=hfunc(data(:,1),data(:,2),theta{1,1},CopulaSpec);
for k=1:N-3
    u{1,2*k}=hfunc(data(:,k+2),data(:,k+1),theta{1,k+1},CopulaSpec);
    u{1,2*k+1}=hfunc(data(:,k+1),data(:,k+2),theta{1,k+1},CopulaSpec);
end
u{1,2*N-4}=hfunc(data(:,N),data(:,N-1),theta{1,N-1},CopulaSpec);
for j=2:(N-1)
    fprintf(1,'computing copula parameters for level %d\n',j)
    for i=1:(N-j)
        [theta{j,i},lik(j,i)]=fitCopula([u{j-1,2*i-1} u{j-1,2*i}],CopulaSpec);
    end
    if j<N-1
    u{j,1}=hfunc(u{j-1,1},u{j-1,2},theta{j,1},CopulaSpec); 
    if N>4
        for i=1:(N-j-2)
        u{j,2*i}=hfunc(u{j-1,2*i+2},u{j-1,2*i+1},theta{j,i+1},CopulaSpec);
        u{j,2*i+1}=hfunc(u{j-1,2*i+1},u{j-1,2*i+2},theta{j,i+1},CopulaSpec);
        end
    end
    u{j,2*N-2*j-2}=hfunc(u{j-1,2*N-2*j},u{j-1,2*N-2*j-1},theta{j,N-j},CopulaSpec);
    end
end
theta=icrIPmat(theta);

end
if  strcmp(CopulaSpec.optimizer,'fmincon')==1
if  strcmp(CopulaSpec.type,'t')==1 
    uncparams=log(theta - 2.01);
else
    uncparams=log((theta-.00009)/.85);
end
elseif strcmp(CopulaSpec.optimizer,'fminunc')==1
uncparams=theta;
if  strcmp(CopulaSpec.type,'t')==1 
    theta=2.01+exp(theta);
else
    theta=.0001+.85./(1+exp(-theta));
end
end
logL=sum(sum(lik));
toc
