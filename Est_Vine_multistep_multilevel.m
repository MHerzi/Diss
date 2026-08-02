function [StrOutput,phi] = Est_Vine_multistep_multilevel(CopulaSpec,N,data,ParSize,T)

% Helper function for fitCopulaVine_multistep_grm
% Estimates multistep D-Vine with different dynamic specifiactions
%
% Author: Martin Grziska
% last modification: 01/13/2011

% start time
% tic;
% -------------------------- options --------------------------------------
options = optimset('fmincon');
options = optimset(options, 'Algorithm','trust-region-reflective','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
    'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);
% --------------------- upper and lower bounds ---------------------------

[lower,upper,defvals] = VineBounds(data,CopulaSpec,1);

% ----------------- the starting value (menu) function --------------------

[startinvals,A,b,nonlincon] = VineStartinvals(CopulaSpec,1,defvals,N,T,data);

%
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

%     Estimation of first D-Vine level
% put data into cell
v=cell(N,N);
phi = cell(N-1,N-1);
likhood = cell(N-1,N-1);
exitflag = cell(N-1,N-1);
output = cell(N-1,N-1);
lambda = cell(N-1,N-1);
grad = cell(N-1,N-1);
hessian = cell(N-1,N-1);
Rt = cell(N-1,N-1);
LogL = cell(N-1,N-1);
likeli = cell(N-1,N-1);

for i=1:N
    v{1,i} = data(:,i);
end

epsilon = 10^(-3); %setze Epsilon auf diesen Standard

% %         if strcmp(CopulaSpec.derivatives,'on')==1
g=1;
for i=1:N-1
    % fot t-copula startinvals of first level are estimated seperatly
    if strcmp(CopulaSpec.type,'t')
        if strcmp(CopulaSpec.corrspec,'DCC') || strcmp(CopulaSpec.corrspec,'TVC')
            startinvals_new = startinvals(g:g+2);
            g=g+3;
        elseif strcmp(CopulaSpec.corrspec,'ADCC')
            startinvals_new = startinvals(g:g+3);
            g=g+4;
        elseif strcmp(CopulaSpec.corrspec,'GDCC')
            startinvals_new = startinvals(g:g+4);
            g=g+5;
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            startinvals_new = startinvals(g:g+6);
            g=g+7;
        elseif strcmp(CopulaSpec.corrspec,'Patton')
            startinvals_new=startinvals;
        end
    else
        if strcmp(CopulaSpec.corrspec,'DCC') || strcmp(CopulaSpec.corrspec,'TVC')
            startinvals_new = startinvals(g:g+1);
            g=g+2;
        elseif strcmp(CopulaSpec.corrspec,'ADCC')
            startinvals_new = startinvals(g:g+2);
            g=g+3;
        elseif strcmp(CopulaSpec.corrspec,'GDCC')
            startinvals_new = startinvals(g:g+3);
            g=g+4;
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            startinvals_new = startinvals(g:g+5);
            g=g+6;
        elseif strcmp(CopulaSpec.corrspec,'Patton')
            startinvals_new=startinvals;
        end
    end
%     teste alle möglichen bivariaten Copulas durch und wähle die optimale
%     nach dem BIC
    [phi{1,i}, likhood{1,i}, exitflag{1,i}, output{1,i}, lambda{1,i}, grad{1,i}, hessian{1,i}] = fmincon('CopulaLL_grm',startinvals_new,A,b,[],[],lower,upper,nonlincon,options,[v{1,i} v{1,i+1}],CopulaSpec,epsilon);
    [LogL{1,i},Rt{1,i},likeli{1,i}] = CopulaLL_grm(phi{1,i},[v{1,i} v{1,i+1}],CopulaSpec);
    constraint=10^-10;
    if exitflag{1,i} <0 || LogL{1,i} == 10E+7      
        epsilon = 10^(-3);
        exitFlag = 0;
        count = 0;
        while (exitFlag < 1 || LogL{1,i} == 10E+7) && count <= 6
            if epsilon < 10^(-3)
                disp(['epsilon = ', num2str(epsilon)])
            end
            options = optimset('fmincon');
            options = optimset(options,'Display','iter','TolCon',constraint,'TolFun',constraint,'TolX',constraint,'Algorithm','interior-point','FunValCheck','on','MaxFunEvals',10000);
            [phi{1,i}, likhood{1,i}, exitflag{1,i}, output{1,i}, lambda{1,i}, grad{1,i}, hessian{1,i}] = fmincon('CopulaLL_grm',startinvals_new,A,b,[],[],lower,upper,nonlincon,options,[v{1,i} v{1,i+1}],CopulaSpec,epsilon);
            [LogL{1,i},Rt{1,i},likeli{1,i}] = CopulaLL_grm(phi{1,i},[v{1,i} v{1,i+1}],CopulaSpec);
            epsilon = epsilon/2;
            count = count + 1;
        end
    end
    %     set back to original options
    options = optimset('fmincon');
    options = optimset(options, 'Algorithm','trust-region-reflective','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
        'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);
end
%     Calculation of conditional Elements (see Aas et al, p.14)
% !!! the rows for v in Aas et al start with 0, here they start with 1, so
%     add 1 to the original code in Aas et al for every row-element of v !!!
if (strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 ...
        || strcmp(CopulaSpec.corrspec,'GDCC') == 1 || strcmp(CopulaSpec.corrspec,'AGDCC') == 1)
    v{2,1}=hfunc_grm(v{1,1},v{1,2},[phi{1,1}],CopulaSpec);
else
    if strcmp(CopulaSpec.type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
        v{2,1}=hfunc_grm(v{1,1},v{1,2},phi{1,1},CopulaSpec);
    elseif strcmp(CopulaSpec.type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 0
        v{2,1}=hfunc_grm(v{1,1},v{1,2},phi{1,1},CopulaSpec);
    end
end
for k=1:N-3
    v{2,2*k}=hfunc_grm(v{1,k+2},v{1,k+1},phi{1,k+1},CopulaSpec);
    v{2,2*k+1}=hfunc_grm(v{1,k+1},v{1,k+2},phi{1,k+1},CopulaSpec);
end
v{2,2*N-4}=hfunc_grm(v{1,N},v{1,N-1},phi{1,N-1},CopulaSpec);
constraint=10^(-5);
for j=2:N-1
    for i=1:N-j
        [startinvals2,A2,b2,nonlincon2] = VineStartinvals2(CopulaSpec,1,defvals,2,T,[v{j-1+1,2*i-1} v{j-1+1,2*i}]);
        [phi{j,i}, likhood{j,i}, exitflag{j,i}, output{j,i}, lambda{j,i}, grad{j,i}, hessian{j,i}] = fmincon('CopulaLL_grm',startinvals2,A2,b2,[],[],lower,upper,nonlincon2,options,[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec,epsilon);
        [LogL{j,i},Rt{j,i},likeli{j,i}] = CopulaLL_grm(phi{j,i},[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec);
        if (exitflag{j,i} < 0|| LogL{j,i} == 10E+7)
            options = optimset('fmincon');
            options = optimset(options,'Display','iter','TolCon',constraint,'TolFun',constraint,'TolX',constraint,'Algorithm','interior-point','FunValCheck','on','MaxFunEvals',10000);
            epsilon = 10^(-3);
            exitFlag = 0;
            count = 0;
            while (exitFlag < 1 || LogL == 10E+7) && count <= 6
                if epsilon < 10^(-3)
                    disp(['epsilon = ', num2str(epsilon)])
                end
                [phi{j,i}, likhood{j,i}, exitflag{j,i}, output{j,i}, lambda{j,i}, grad{j,i}, hessian{j,i}] = fmincon('CopulaLL_grm',startinvals2,A2,b2,[],[],lower,upper,nonlincon2,options,[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec,epsilon);
                [LogL{j,i},Rt{j,i},likeli{j,i}] = CopulaLL_grm(phi{j,i},[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec);
                epsilon = epsilon/2;
                constraint = constraint/2;
                count = count + 1;
            end
        end
        %         set original constraints
        options = optimset('fmincon');
        options = optimset(options, 'Algorithm','trust-region-reflective','Display','iter','MaxIter',500,'MaxFunEvals',10000,...
            'TolCon',10^-6,'TolFun',10^-2,'TolX',10^-6, 'AlwaysHonorConstraints', 'bounds', 'InitBarrierParam', 0.1);
    end
    if j<N-1
        v{j+1,1}=hfunc_grm(v{j-1+1,1},v{j-1+1,2},phi{j,1},CopulaSpec);
        if N>4
            for i=1:(N-2-j)
                v{j+1,2*i} = hfunc_grm(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1},CopulaSpec);
                v{j+1,2*i+1} = hfunc_grm(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1},CopulaSpec);
            end
        end
    end
    if (2*N-2*j-2) > 0
        v{j+1,2*N-2*j-2} = hfunc_grm(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j},CopulaSpec);
    end
end

c=N-1;
likelihood=0;
for i=1:N-1
    for j=1:c
        likelihood = likelihood + LogL{i,j};
    end
    c=c-1;
end
% Generate Output
StrOutput.VineParams = phi;
StrOutput.Hessian=hessian;
StrOutput.Gradient=grad;
StrOutput.exitflag=exitflag;
% StrOutput.timeinseconds=toc;
[AIC, BIC] = aicbic(-likelihood,size(phi{1,1},1)*ParSize,T);
StrOutput.AIC = AIC;
StrOutput.BIC = BIC;
StrOutput.LogL = -likelihood;
StrOutput.Rt = Rt;

% elseif strcmp(CopulaSpec.derivatives,'on')==0
% g=1;
% for i=1:N-1
%     % fot t-copula startinvals of first level are estimated seperatly
%     if strcmp(CopulaSpec.type,'t')
%         if strcmp(CopulaSpec.corrspec,'DCC') || strcmp(CopulaSpec.corrspec,'TVC')
%             startinvals_new = startinvals(g:g+2);
%             g=g+3;
%         elseif strcmp(CopulaSpec.corrspec,'ADCC')
%             startinvals_new = startinvals(g:g+3);
%             g=g+4;
%         elseif strcmp(CopulaSpec.corrspec,'GDCC')
%             startinvals_new = startinvals(g:g+4);
%             g=g+5;
%         elseif strcmp(CopulaSpec.corrspec,'AGDCC')
%             startinvals_new = startinvals(g:g+6);
%             g=g+7;
%         end
%     else
%         if strcmp(CopulaSpec.corrspec,'DCC') || strcmp(CopulaSpec.corrspec,'TVC')
%             startinvals_new = startinvals(g:g+1);
%             g=g+2;
%         elseif strcmp(CopulaSpec.corrspec,'ADCC')
%             startinvals_new = startinvals(g:g+2);
%             g=g+3;
%         elseif strcmp(CopulaSpec.corrspec,'GDCC')
%             startinvals_new = startinvals(g:g+3);
%             g=g+4;
%         elseif strcmp(CopulaSpec.corrspec,'AGDCC')
%             startinvals_new = startinvals(g:g+5);
%             g=g+6;
%         elseif strcmp(CopulaSpec.corrspec,'Patton')
%             startinvals_new=startinvals;
%         end
%     end
%     [phi{1,i}, likhood{1,i}, exitflag{1,i}] = fmincon('CopulaLL_grm',startinvals_new,A,b,[],[],lower,upper,nonlincon,options,[v{1,i} v{1,i+1}],CopulaSpec,epsilon);
%     [LogL{1,i},Rt{1,i},likeli{1,i}] = CopulaLL_grm(phi{1,i},[v{1,i} v{1,i+1}],CopulaSpec);
%     if exitflag{1,i} < 0
%         options = optimset('fmincon');
%         options = optimset(options,'Display','iter','TolCon',10^-20,'TolFun',10^-4,'TolX',10^-6,'Algorithm','interior-point','FunValCheck','on','MaxFunEvals',10000);
%         epsilon = 10^(-3);
%         exitFlag = 0;
%         count = 0;
%         while exitFlag < 1 && count <= 10
%             if epsilon < 10^(-3)
%                 disp(['epsilon = ', num2str(epsilon)])
%             end
%             [phi{1,i}, likhood{1,i}, exitflag{1,i}] = fmincon('CopulaLL_grm',startinvals_new,A,b,[],[],lower,upper,nonlincon,options,[v{1,i} v{1,i+1}],CopulaSpec,epsilon);
%             [LogL{1,i},Rt{1,i},likeli{1,i}] = CopulaLL_grm(phi{1,i},[v{1,i} v{1,i+1}],CopulaSpec);
%             epsilon = epsilon/2;
%             count = count + 1;
%         end
%     end
% end
% %     Calculation of conditional Elements (see Aas et al, p.14)
% % !!! the rows for v in Aas et al start with 0, here they start with 1, so
% %     add 1 to the original code in Aas et al for every row-element of v !!!
% if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 ...
%         || strcmp(CopulaSpec.corrspec,'GDCC') == 1 || strcmp(CopulaSpec.corrspec,'AGDCC')
%     v{2,1}=hfunc_grm(v{1,1},v{1,2},[phi{1,1}],CopulaSpec);
% else
%     if strcmp(CopulaSpec.type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
%         v{2,1}=hfunc_grm(v{1,1},v{1,2},phi{1,1},CopulaSpec);
%     elseif strcmp(CopulaSpec.type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 0
%         v{2,1}=hfunc_grm(v{1,1},v{1,2},phi(1,1),CopulaSpec);
%     end
% end
%
% for k=1:N-3
%     v{2,2*k}=hfunc_grm(v{1,k+2},v{1,k+1},phi{1,k+1},CopulaSpec);
%     v{2,2*k+1}=hfunc_grm(v{1,k+1},v{1,k+2},phi{1,k+1},CopulaSpec);
% end
% v{2,2*N-4}=hfunc_grm(v{1,N},v{1,N-1},phi{1,N-1},CopulaSpec);
% for j=2:N-1
%     for i=1:N-j
%         [startinvals2,A2,b2,nonlincon2] = VineStartinvals(CopulaSpec,1,defvals,2,T,[v{j-1+1,2*i-1} v{j-1+1,2*i}]);
%         [phi{j,i}, likhood{j,i}, exitflag{j,i}] = fmincon('CopulaLL_grm',startinvals2,A2,b2,[],[],lower,upper,nonlincon2,options,[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec,epsilon);
%         [LogL{j,i},Rt{j,i},likeli{j,i}] = CopulaLL_grm(phi{j,i},[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec,epsilon);
%         if exitflag{j,i} < 0
%             options = optimset('fmincon');
%             options = optimset(options,'Display','iter','TolCon',10^-20,'TolFun',10^-15,'TolX',10^-15,'Algorithm','interior-point','FunValCheck','on','MaxFunEvals',10000);
%             epsilon = 10^(-3);
%             exitFlag = 0;
%             count = 0;
%             while exitFlag < 1 && count <= 10
%                 if epsilon < 10^(-3)
%                     disp(['epsilon = ', num2str(epsilon)])
%                 end
%                 [phi{j,i}, likhood{j,i}, exitflag{j,i}] = fmincon('CopulaLL_grm',startinvals2,A2,b2,[],[],lower,upper,nonlincon2,options,[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec,epsilon);
%                 [LogL{j,i},Rt{j,i},likeli{j,i}] = CopulaLL_grm(phi{j,i},[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec,epsilon);
%                 epsilon = epsilon/2;
%                 count = count + 1;
%             end
%         end
%     end
%     if j<N-1
%         v{j+1,1}=hfunc_grm(v{j-1+1,1},v{j-1+1,2},phi{j,1},CopulaSpec);
%         if N>4
%             for i=1:(N-2-j)
%                 v{j+1,2*i} = hfunc_grm(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1},CopulaSpec);
%                 v{j+1,2*i+1} = hfunc_grm(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1},CopulaSpec);
%             end
%         end
%     end
%     if (2*N-2*j-2) > 0
%         v{j+1,2*N-2*j-2} = hfunc_grm(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j},CopulaSpec);
%     end
% end
% %     calculate complete likelihood (see also Aas)
% c=N-1;
% likelihood=0;
% for i=1:N-1
%     for j=1:c
%         likelihood = likelihood + LogL{i,j};
%     end
%     c=c-1;
% end
% % Generate Output
% StrOutput.VineParams = phi;
% StrOutput.Hessian=hessian;
% StrOutput.Gradient=grad;
% StrOutput.exitflag=exitflag;
% % StrOutput.timeinseconds=toc;
% [AIC, BIC] = aicbic(-likelihood,size(phi{1,1},1)*ParSize,T);
% StrOutput.AIC = AIC;
% StrOutput.BIC = BIC;
% StrOutput.LogL = -likelihood;
% StrOutput.Rt = Rt;



% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Code für unconstrainded Optimization (kann mit den verschiedenen
% DCC-Spezifikationen nicht mehr benutzt werden
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% elseif strcmp(CopulaSpec.optimizer,'fminunc')
%     if strcmp(CopulaSpec.derivatives,'on')==1
%         g=1;
%         for i=1:N-1
%             % fot t-copula startinvals of first level are estimated seperatly
%             if strcmp(CopulaSpec.type,'t')
%                 startinvals_new = startinvals(g:g+2);
%                 g=g+3;
%             else
%                 startinvals_new=startinvals;
%             end
%             [phi{1,i}, likhood{1,i}, exitflag{1,i}, output{1,i}, lambda{1,i}, grad{1,i}, hessian{1,i}] = fminunc('CopulaLL_grm',startinvals_new,options,[v{1,i} v{1,i+1}],CopulaSpec);
%             [LogL{1,i},Rt{1,i},likeli{1,i}] = CopulaLL_grm(phi{1,i},[v{1,i} v{1,i+1}],CopulaSpec);
%         end
%         %     Calculation of conditional Elements (see Aas et al, p.14)
%         % !!! the rows for v in Aas et al start with 0, here they start with 1, so
%         %     add 1 to the original code in Aas et al for every row-element of v !!!
%         if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 ...
%                 || strcmp(CopulaSpec.corrspec,'GDCC') == 1 || strcmp(CopulaSpec.corrspec,'AGDCC') == 1
%             v{2,1}=hfunc_grm(v{1,1},v{1,2},[phi{1,1}],CopulaSpec);
%         else
%             if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
%                 v{2,1}=hfunc_grm(v{1,1},v{1,2},phi{1,1},CopulaSpec);
%             else
%                 v{2,1}=hfunc_grm(v{1,1},v{1,2},phi(1,1),CopulaSpec);
%             end
%         end
%
%         for k=1:N-3
%             v{2,2*k}=hfunc_grm(v{1,k+2},v{1,k+1},phi{1,k+1},CopulaSpec);
%             v{2,2*k+1}=hfunc_grm(v{1,k+1},v{1,k+2},phi{1,k+1},CopulaSpec);
%         end
%
%         v{2,2*N-4}=hfunc_grm(v{1,N},v{1,N-1},phi{1,N-1},CopulaSpec);
%
%         %     create new startinvals for all levels but the first
%         [startinvals,A,b,nonlincon] = VineStartinvals(CopulaSpec,N*(N-1)*.5,defvals,N,T,data);
%         if strcmp(CopulaSpec.type,'t')
%             startinvals = startinvals(end-2:end);
%         end
%         for j=2:N-1
%             for i=1:N-j
%                 [phi{j,i}, likhood{j,i}, exitflag{j,i}, output{j,i}, lambda{j,i}, grad{j,i}, hessian{j,i}] = fmincon('CopulaLL_grm',startinvals,A,b,[],[],lower,upper,nonlincon,options,[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec);
%                 [LogL{j,i},Rt{j,i},likeli{j,i}] = CopulaLL_grm(phi{j,i},[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec);
%             end
%             if j<N-1
%                 v{j+1,1}=hfunc_grm(v{j-1+1,1},v{j-1+1,2},phi{j,1},CopulaSpec);
%                 if N>4
%                     for i=1:(N-2-j)
%                         v{j+1,2*i} = hfunc_grm(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1},CopulaSpec);
%                         v{j+1,2*i+1} = hfunc_grm(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1},CopulaSpec);
%                     end
%                 end
%             end
%             if (2*N-2*j-2) > 0
%                 v{j+1,2*N-2*j-2} = hfunc_grm(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j},CopulaSpec);
%             end
%         end
%     elseif strcmp(CopulaSpec.derivatives,'on')==0
%         g=1;
%         for i=1:N-1
%             % fot t-copula startinvals of first level are estimated seperatly
%             if strcmp(CopulaSpec.type,'t')
%                 startinvals_new = startinvals(g:g+2);
%                 g=g+3;
%             else
%                 startinvals_new = startinvals;
%             end
%             [phi{1,i}, likhood{1,i}, exitflag{1,i}] = fminunc('CopulaLL_grm',startinvals_new,options,[v{1,i} v{1,i+1}],CopulaSpec);
%             [LogL{1,i},Rt{1,i},likeli{1,i}] = CopulaLL_grm(phi{1,i},[v{1,i} v{1,i+1}],CopulaSpec);
%         end
%         %     Calculation of conditional Elements (see Aas et al, p.14)
%         % !!! the rows for v in Aas et al start with 0, here they start with 1, so
%         %     add 1 to the original code in Aas et al for every row-element of v !!!
%         if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
%             v{2,1}=hfunc_grm(v{1,1},v{1,2},[phi{1,1}],CopulaSpec);
%         else
%             if strcmp(CopulaSpec.type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
%                 v{2,1}=hfunc_grm(v{1,1},v{1,2},phi{1,1},CopulaSpec);
%             else
%                 v{2,1}=hfunc_grm(v{1,1},v{1,2},phi(1,1),CopulaSpec);
%             end
%         end
%
%         for k=1:N-3
%             v{2,2*k}=hfunc_grm(v{1,k+2},v{1,k+1},phi{1,k+1},CopulaSpec);
%             v{2,2*k+1}=hfunc_grm(v{1,k+1},v{1,k+2},phi{1,k+1},CopulaSpec);
%         end
%
%         v{2,2*N-4}=hfunc_grm(v{1,N},v{1,N-1},phi{1,N-1},CopulaSpec);
%
%         %     create new startinvals for all levels but the first
%         [startinvals] = VineStartinvals(CopulaSpec,1,defvals,N,T,data);
%         if strcmp(CopulaSpec.type,'t')
%             startinvals = startinvals(end-2:end);
%             startinvals(end) = 10;
%         end
%         for j=2:N-1
%             for i=1:N-j
%                 [phi{j,i}, likhood{j,i}, exitflag{j,i}] = fminunc('CopulaLL_grm',startinvals,options,[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec);
%                 [LogL{j,i},Rt{j,i},likeli{j,i}] = CopulaLL_grm(phi{1,i},[v{j-1+1,2*i-1} v{j-1+1,2*i}],CopulaSpec);
%             end
%             if j<N-1
%                 v{j+1,1}=hfunc_grm(v{j-1+1,1},v{j-1+1,2},phi{j,1},CopulaSpec);
%                 if N>4
%                     for i=1:(N-2-j)
%                         v{j+1,2*i} = hfunc_grm(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1},CopulaSpec);
%                         v{j+1,2*i+1} = hfunc_grm(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1},CopulaSpec);
%                     end
%                 end
%             end
%             if (2*N-2*j-2) > 0
%                 v{j+1,2*N-2*j-2} = hfunc_grm(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j},CopulaSpec);
%             end
%         end
%     end
% end

% %     calculate complete likelihood (see also Aas)
% c=N-1;
% likelihood=0;
% for i=1:N-1
%     for j=1:c
%         likelihood = likelihood + LogL{i,j};
%     end
%     c=c-1;
% end
% % Generate Output
% StrOutput.VineParams = phi;
% StrOutput.Hessian=hessian;
% StrOutput.Gradient=grad;
% StrOutput.exitflag=exitflag;
% % StrOutput.timeinseconds=toc;
% [AIC, BIC] = aicbic(-likelihood,size(phi{1,1},1)*ParSize,T);
% StrOutput.AIC = AIC;
% StrOutput.BIC = BIC;
% StrOutput.LogL = -likelihood;
% StrOutput.Rt = Rt;