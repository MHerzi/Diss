function [lower, upper, defvals] = VineBounds(data,CopulaSpec,ParSize)

% Helper function for fitCopulaVine_grm.m. Claculates upper bounds, lower
% bounds, and default starting values.
%
% USAGE:
%        [lower, upper, defvals] = VineBounds(data,CopulaSpec,ParSize)
%
% INPUTS:
%        data:       t x k array fof unif(0,1) variables
%  CopulaSpec:       structure from setCopulaVineLLinputs.m
%     ParSize:       # of (conditional) bivariate copulas to be estimated
%                    in vine

[T,N] = size(data);
type = CopulaSpec.type;
if strcmp(CopulaSpec.optimizer,'fmincon')
    if strcmp(type,'t') == 1
        if strcmp(CopulaSpec.corrspec,'static') == 1
            %             bounds for multistep
            %             static t-needs starting values for d.o.f and
            %             correlation parameters; unconditional
            %             correlation parameters are used
            %                 correlation estimated with kendall's tau, makes
            %                 comparsion with dynamic models possible
            tau=corr(data,'type','kendall');
            rho=sin(tau*pi/2);
            h=1;
            c=1;
            for i=2:size(rho,2)
                for j=1:c
                    rho1(h) = rho(i,j);
                    h=h+1;
                end
                c=c+1;
            end
            defvals = [10*ones(ParSize,1) rho1'] ;
            lower = [2.1*ones(ParSize,1) -1*ones(ParSize,1)+1e-6];
            upper = [300*ones(ParSize,1) ones(ParSize,1)-1e-6];
        elseif strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1
            %              bounds for one-step estimation
            %                 starting values for d.o.f
            %             defvals = 10*ones(ParSize*3,1);
            %             %                dynamic constraint: a+b<1
            %             for i=1:3:(ParSize*3)
            %                 lower(i:i+2) = [0+1e-6 0+1e-6 2.1];
            %             end
            %             for i=1:3:(ParSize*3)
            %                 upper(i:i+2) = [.99999 .99999 300];
            %             end
            %
            % bounds for multistep
            defvals = 10;
            %             lower = [0+1e-6 0+1e-6 2.1];
            lower = [0+1e-6 0+1e-6 2.1];
            % lower=[0+1e-6 0.2657];
            upper = [1-1e-6 1-1e-6 300];
        elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
            % %             bounds for single step
            %             for i=1:3:(ParSize*4)
            %                 lower(i:i+3) = [0+1e-6 0+1e-6 0+1e-6 2.1];
            %             end
            %             for i=1:3:(ParSize*4)
            %                 upper(i:i+3) = [.9999 .9 .9999 300];
            %             end
            %             defvals=10*ones(ParSize*4,1);
            %         end
            
            % bounds for multistep
            lower = [zeros(1,3)+1e-6 2.1];
            %             lower = [zeros(1,2)+1e-6 0.2657 2.1];
            upper = [1-1e-6 1-1e-2 1-1e-6 300];
            defvals=1;
        elseif strcmp(CopulaSpec.corrspec,'GDCC')
            lower = [zeros(1,4)+1e-6 2.1];
            upper = [ones(1,4)-1e-6 300];
            defvals = 1;
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            lower = [zeros(1,6)+1e-6 2.1];
            upper = [ones(1,6)-1e-6 300];
            defvals = 1;
        end
        
    elseif strcmp(type,'Clayton') == 1 || strcmp(type,'Rotated Clayton')
        if strcmp(CopulaSpec.corrspec,'static') == 1
            defvals=ones(ParSize,1);
            lower = zeros(ParSize,1)+1e-6;
            upper = 50;
        elseif strcmp(CopulaSpec.corrspec,'Patton') == 1
            %                 defvals=.15*ones(ParSize,1);
            defvals = .15;
            lower =[];
            upper =[];
        elseif strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1
            % %             bounds for single-step
            %             defvals=ones(ParSize*2,1);
            %             for i=1:2:(ParSize*2)
            %                 lower(i:i+1) = [0+1e-5 0+1e-5];
            %             end
            %             for i=1:2:(ParSize*2)
            %                 upper(i:i+1) = [1-1e-6 1-1e-6];
            %             end
            
            % bounds for multistep
            defvals=1;
            lower= [0+1e-6 0+1e-6];
            %             lower = [0+1e-6 0.2657];
            upper =  [1-1e-6 1-1e-6];
        elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
            %             bounds for single-step
            %             defvals=ones(ParSize*3,1);
            %             for i=1:3:(ParSize*3)
            %                 lower(i:i+2) = [0+1e-6 0+1e-6 0+1e-6];
            %             end
            %             for i=1:3:(ParSize*3)
            %                 upper(i:i+2) = [.9999 .9 .9999];
            %             end
            
            % bounds for multistep
            defvals=1;
            lower = [0+1e-6 0+1e-6 0+1e-6];
            %             lower = [0+1e-6 0+1e-6 0.2657];
            upper = [1-1e-6 .9 1-1e-6];
        elseif strcmp(CopulaSpec.corrspec,'GDCC')
            lower = zeros(1,4)+1e-6;
            upper = ones(1,4)-1e-6;
            defvals = 1;
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            lower = zeros(1,6)+1e-6;
            upper = ones(1,6)-1e-6;
            defvals = 1;
        end
    elseif strcmp(type,'SJC') == 1
        display('the copula parameters is a .5*N*(N-1)x2 vector');
        defvals=.15*ones(ParSize,2);
        lower =10^-4*ones(ParSize,2);
        upper =.65*ones(ParSize,2);
    elseif strcmp(type,'Gaussian') == 1
        if (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC')) == 1
            % %                 bounds for single-step
            %                 for i=1:2:(ParSize*2)
            %                     lower(i:i+1) = [0+1e-10 0+1e-10];
            %                 end
            %                 for i=1:2:(ParSize*2)
            %                     upper(i:i+1) = [.9999 .9999];
            %                 end
            % bounds for multistep
            lower = [0+1e-6 0+1e-6];
            %             lower = [0+1e-6 0.2657];
            upper = [1-1e-6 1-1e-6];
        elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
            %                 bounds for single-step
            %                 for i=1:3:(ParSize*3)
            %                     lower(i:i+2) = [0+1e-6 0+1e-6 0+1e-6];
            %                 end
            %                 for i=1:3:(ParSize*3)
            %                     upper(i:i+2) = [.9999 .9 .9999];
            %                 end
            
            % bounds for multistep
            lower = [0+1e-6 0+1e-6 0+1e-6];
            %             lower = [0+1e-6 0+1e-6 0.2657];
            upper = [1-1e-6 .9 1-1e-6];
        elseif strcmp(CopulaSpec.corrspec,'static') == 1
            lower = -1;
            upper = 1;
        elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
            lower = zeros(1,4)+1e-6;
            upper = ones(1,4)-1e-6;
        elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
            lower = zeros(1,6)+1e-6;
            upper = ones(1,6)-1e-6;
        end
        defvals=[];
    elseif strcmp(type,'Gumbel') == 1
        if strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1
            % %                 bounds for single-step
            %                 defvals = 2*ones(ParSize*2,1);
            %                 for i=1:2:(ParSize*2)
            %                     lower(i:i+1) = [0+1e-6 0+1e-6];
            %                 end
            %                 for i=1:2:(ParSize*2)
            %                     upper(i:i+1) = [.9999 .9999];
            %                 end
            % bounds for multistep
            lower = [1e-6 1e-6;];
            %             lower = [1e-6 0.2657];
            upper =  [1-1e-6 1-1e-6];
            defvals= 2;
        elseif  strcmp(CopulaSpec.corrspec,'ADCC') == 1
            % %                 bounds for single-step
            %                 defvals = 2*ones(ParSize*3,1);
            %                 for i=1:3:(ParSize*3)
            %                     lower(i:i+2) = [0+1e-6 0+1e-6 0+1e-6];
            %                 end
            %                 for i=1:3:(ParSize*3)
            %                     upper(i:i+2) = [.9999 .9 .9999];
            %                 end
            
            % bounds for multistep
            lower = [1e-6 1e-6 1e-6];
            lower = [1e-6 1e-6 0.2657];
            upper = [1-1e-6 .9 1-1e-6];
            defvals=2;
        elseif strcmp(CopulaSpec.corrspec,'GDCC')
            lower = zeros(1,4)+1e-6;
            upper = ones(1,4)*.9999;
            defvals = 2;
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            lower = zeros(1,6)+1e-6;
            upper = ones(1,6)*.9999;
            defvals = 2;
        elseif strcmp(CopulaSpec.corrspec,'Patton') == 1
            defvals=[.7 .1 .1];
            lower = -100*ones(3,1);  % theoretisch keine Beschränkungen, aber hilft manchmal bei Optimierung
            upper =  100*ones(3,1);
        elseif strcmp(CopulaSpec.corrspec,'static') == 1
            lower = repmat(1.05,ParSize,1);
            upper = 20;
            defvals = repmat(2,ParSize,1);
        end
    elseif strcmp(CopulaSpec.optimizer,'fminunc')
        if strcmp(type,'t')
            display('the copula parameters is a .5*N*(N-1)x1 vector');
            defvals=2.8898*ones(.5*N*(N-1),1);
        elseif strcmp(type,'Clayton') || strcmp(type,'Rotated Clayton')
            display('the copula parameters is a .5*N*(N-1)x1 vector');
            defvals=-1.9726*ones(.5*N*(N-1),1);
        elseif strcmp(type,'SJC')
            display('the copula parameters is a .5*N*(N-1)x2 vector');
            defvals=-5.295*ones(.5*N*(N-1),2);
        elseif strcmp(type,'Gumbel')
            display('the copula parameters is a .5*N*(N-1)x2 vector');
            defvals=-1.9726*ones(.5*N*(N-1),1);
        end
        lower=[];
        upper=[];
    end
end