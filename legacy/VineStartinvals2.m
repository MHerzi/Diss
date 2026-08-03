function [startinvals,A,b,nonlincon] = VineStartinvals2(CopulaSpec,ParSize,defvals,N,T,data)
% Helper function for Est_Vine_multistep; estimates startinvals for all
% stages but the first
%
% INPUTS:
%        CopulaSpec : Specification structure from setCopulaVineInputs
%        ParSize    : 0.5*N*(N-1)
%        defvals    : default values for startinvals
%        N          : N of TxN data matrix
%        data       : TxN data matrix
%        level      : 1 for first level decomposition of multistep vine, else 2
%
% OUTPUTS:
%         startinvals : vector of startinvals
%         A           : matrix for fminon constraints: A*startinvals <= b
%         b           : matrix for fminon constraints: A*startinvals <= b
%         nonlincon   : string of name of nonlincon for sepcific level
%
% Author: Martin Grziska
% Date of last modofication: 09/10/2010

if strcmp(CopulaSpec.type,'t') == 1
    %             startinvals for bivariate copulas first decomposition
    for i=1:N-1
        [Holder,start_nu(i)] = copulafit('t',data(:,i:i+1));
        if start_nu(i)>100
            start_nu(i) = 10;
        end
    end
    if strcmp(CopulaSpec.corrspec, 'DCC')
        %         schätzte DCC(1,1)
        %         for i=1:N-1
        %             stdresid = tinv(data(:,i:i+1),start_nu(i));
        %             [parameters{i}] = Est_DCC(stdresid,1,1);
        %         end
        for i=1:N-1
            parameters{i}=[.1 .8]';
        end
        %         Form der Startinvals: [a; b; nu]]
        %                 1.startinval for first bivariate copula
        startinvals = [parameters{1}; start_nu(i)];
        %                 startinvals for rest of bivariate copulas first
        %                 decomposition
        for i=2:N-1
            startinvals = [startinvals;[parameters{i}; start_nu(i)]];
        end
        %                 startinvals for conditional bivariate copulas
        A=[];
        b=[];
        nonlincon='Nonlincon_vine';
    elseif strcmp(CopulaSpec.corrspec, 'GDCC')
        %         bestimme startinvals mit DCC-GARCH Schätzung
        %         for i=1:N-1
        %             stdresid = tinv(data(:,i:i+1),start_nu(i));
        %             [parameters{i}] = Est_GDCC(stdresid,1,1);
        %         end
        
        for i=1:N-1
            parameters{i}=[.1 .1 .8 .8]';
        end
        %         startinvals für erste bivariate Copula
        startinvals = [parameters{1}; start_nu(1)];
        %         startinvals für restliche Variablen erste Ebene
        for i=2:N-1
            startinvals = [startinvals;[parameters{i}; start_nu(i)]]; %parameters
        end
        A=[];
        b=[];
        nonlincon='Nonlincon_vine';
    elseif strcmp(CopulaSpec.corrspec, 'AGDCC')
        %         for i=1:N-1
        %             stdresid = norminv(data(:,i:i+1)); %DCC GARCH-Modelle sind nur für Normalverteilung spezifiziert
        %             [parameters{i}] = Est_AGDCC(stdresid,1,1);
        %         end
        for i=1:N-1
            parameters{i}=[.1 .1 .01 .01 .8 .8'];
        end
        %         startinvals für erste bivariate Copula
        startinvals = [parameters{1}; start_nu(1)];
        %         startinvals für restliche Variablen erste Ebene
        for i=2:N-1
            startinvals = [startinvals;[parameters{i}; start_nu(i)]]; %parameters
        end
        A=[];
        b=[];
        nonlincon='Nonlincon_vine';
    elseif strcmp(CopulaSpec.corrspec, 'TVC') == 1
        startinvals = repmat([.05; .9; 15],ParSize,1);
        %         kreiere spezielle Matrix A, so das A*x<b
        A = zeros(ParSize*3,ParSize*3);
        k=1;
        for i = 1:3:ParSize*3
            A(i,k:k+1)=[1 1];
            k=k+3;
        end
        b = repmat([1-(1e-10); 0; 0],ParSize,1);
        nonlincon = [];
    elseif strcmp(CopulaSpec.corrspec, 'ADCC')
        %         for i=1:N-1
        %             stdresid = tinv(data(:,i:i+1),start_nu(i));
        %             [parameters{i}] = Est_ADCC(stdresid,1,1);
        %         end
        
        for i=1:N-1
            parameters{i}=[.1 .01 .8]';
        end
        %         startinvals für erste bivariate Copula
        startinvals = [parameters{1}; start_nu(1)];
        %         startinvals für restliche Variablen erste Ebene
        for i=2:N-1
            startinvals = [startinvals;[parameters{i}; start_nu(i)]]; %parameters
        end
        A=[];
        b=[];
        nonlincon = 'Nonlincon_vine';
    else
        startinvals=defvals;
        A=[];
        b=[];
        nonlincon = [];
    end
elseif strcmp(CopulaSpec.type,'Gaussian') || strcmp(CopulaSpec.type,'Clayton') || strcmp(CopulaSpec.type,'Rotated Clayton') || strcmp(CopulaSpec.type,'Gumbel')
    %             startinvals for bivariate copulas first decomposition
    if strcmp(CopulaSpec.corrspec, 'DCC')
        %         schätzte DCC(1,1)
        %         for i=1:N-1
        %             stdresid = norminv(data(:,i:i+1));
        %             [parameters{i}] = Est_DCC(stdresid,1,1);
        %         end
        
        for i=1:N-1
            parameters{i}=[.1 .8]';
        end
        %         Form der Startinvals: [a; b; nu]]
        %                 1.startinval for first bivariate copula
        startinvals = parameters{1};
        %                 startinvals for rest of bivariate copulas first
        %                 decomposition
        for i=2:N-1
            startinvals = [startinvals;parameters{i}];
        end
        %                 startinvals for conditional bivariate copulas
        A=[];
        b=[];
        nonlincon='Nonlincon_vine';
    elseif strcmp(CopulaSpec.corrspec, 'GDCC')
        %         bestimme startinvals mit DCC-GARCH Schätzung
        %         for i=1:N-1
        %             stdresid = norminv(data(:,i:i+1));
        %             [parameters{i}] = Est_GDCC(stdresid,1,1);
        %         end
        
        for i=1:N-1
            parameters{i}=[.1 .1 .8 .8]';
        end
        %         startinvals für erste bivariate Copula
        startinvals = parameters{1};
        %         startinvals für restliche Variablen erste Ebene
        for i=2:N-1
            startinvals = [startinvals;[parameters{i}]]; %parameters
        end
        A=[];
        b=[];
        nonlincon='Nonlincon_vine';
    elseif strcmp(CopulaSpec.corrspec, 'AGDCC')
        %         for i=1:N-1
        %             stdresid = norminv(data(:,i:i+1));
        %             [parameters{i}] = Est_AGDCC(stdresid,1,1);
        %         end
        
        for i=1:N-1
            parameters{i}=[.1 .1 .01 .01 .8 .8]';
        end
        %         startinvals für erste bivariate Copula
        startinvals = parameters{1};
        %         startinvals für restliche Variablen erste Ebene
        for i=2:N-1
            startinvals = [startinvals;parameters{i}]; %parameters
        end
        A=[];
        b=[];
        nonlincon='Nonlincon_vine';
    elseif strcmp(CopulaSpec.corrspec, 'TVC') == 1
        startinvals = repmat([.05; .9; 15],ParSize,1);
        %         kreiere spezielle Matrix A, so das A*x<b
        A = zeros(ParSize*3,ParSize*3);
        k=1;
        for i = 1:3:ParSize*3
            A(i,k:k+1)=[1 1];
            k=k+3;
        end
        b = repmat([1-(1e-10); 0; 0],ParSize,1);
        nonlincon = [];
    elseif strcmp(CopulaSpec.corrspec, 'ADCC')
        %         for i=1:N-1
        %             stdresid = norminv(data(:,i:i+1));
        %             [parameters{i}] = Est_ADCC(stdresid,1,1);
        %         end
        
        for i=1:N-1
            parameters{i}=[.1 .01 .8]';
        end
        %         startinvals für erste bivariate Copula
        startinvals = parameters{1};
        %         startinvals für restliche Variablen erste Ebene
        for i=2:N-1
            startinvals = [startinvals;parameters{i}]; %parameters
        end
        A=[];
        b=[];
        nonlincon = 'Nonlincon_vine';
    elseif strcmp(CopulaSpec.corrspec,'Patton')
        %         startinvals=repmat(defvals,3,1);
        startinvals=defvals;
        A=[];
        b=[];
        nonlincon = [];
    end
else
    startinvals=inputstartinvals_grm(CopulaSpec,defvals);
    A=[];
    b=[];
    nonlincon = [];
end


% -------------------------------------------------------------------------
% -------- alter Code----------------------------------------------------
% if strcmp(CopulaSpec.type,'Clayton') == 1
%     if strcmp(CopulaSpec.corrspec, 'Patton') == 1;
%         %         Form der Startinvals: [omega, alpha; beta]
%         startinvals = repmat([.5; .1; .1],ParSize,1);
%         %         constraint AR-Term < 1; A*x<=b
%         A = zeros(ParSize,size(startinvals,1));
%         g=3;
%         %         setze 1 an die Stelle des AR-Terms
%         for j=1:ParSize
%             A(j,g)=1;
%             g=g+3;
%         end
%         b = ones(ParSize,1)-1e-6;
%         nonlincon = [];
%     elseif strcmp(CopulaSpec.corrspec,'static') == 1
%         startinvals = defvals;
%         nonlincon = [];
%         A=[];
%         b=[];
%     elseif strcmp(CopulaSpec.corrspec,'DCC') == 1;
%         %             Es werden die Parameter der Claytion-Copula geschätzt; die
%         %             DCC-Parameter werden unter der Gauss-Annahme geschätzt und
%         %             dann durch Transformation mittels Kenndals tau in
%         %             Clayton-Parameter übertragen; an dieser Stelle müssen die
%         %             Clayton-Parameter so restringiert werden, dass die
%         %             transformierten Parameter wieder den DCC-Restriktionen
%         %             entsprechen
%         %             startinvals = repmat([.006; 1.9],ParSize,1);
%         startinvals = repmat([.02; .95],ParSize,1);
%         %         A = zeros(ParSize*2,ParSize*2);
%         %         k=1;
%         %         for i = 1:2:ParSize*2
%         %             A(i,k:k+1)=[1 1];
%         %             k=k+2;
%         %         end
%         %         b = repmat([1-1e-5; 0],ParSize,1);
%         A=[];
%         b=[];
%         nonlincon = 'Nonlincon_vine';
%     elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1;
%         startinvals = repmat([.01; .001; .97],ParSize,1);
%         A=[];
%         b=[];
%         nonlincon = 'Nonlincon_vine';
%     elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
%         startinvals = repmat([.01; .97],ParSize,1);
%         %         kreiere spezielle Matrix A, so das A*x<b
%         A = zeros(ParSize*2,ParSize*2);
%         k=1;
%         for i = 1:2:ParSize*2
%             A(i,k:k+1)=[1 1];
%             k=k+2;
%         end
%         b = repmat([1-(1e-10); 0],ParSize,1);
%         nonlincon = [];
%     elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
%         startinvals = repmat([ones(2,1)*.01; ones(2,1)*.97],ParSize,1);
%         A=[];
%         b=[];
%         nonlincon = 'Nonlincon_vine';
%     elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
%         startinvals = repmat([ones(2,1)*.01; ones(2,1)*.0001; ones(2,1)*.97],ParSize,1);
%         A=[];
%         b=[];
%         nonlincon = 'Nonlincon_vine';
%     end

% elseif strcmp(CopulaSpec.type,'Gumbel') == 1
%     if strcmp(CopulaSpec.corrspec, 'DCC')
%         startinvals = repmat([.01; .97],ParSize,1);
%         A = zeros(ParSize*2,ParSize*2);
%         k=1;
%         for i = 1:2:ParSize*2
%             A(i,k:k+1)=[1 1];
%             k=k+2;
%         end
%         b = repmat([1-(1e-10); 0],ParSize,1);
%         nonlincon = [];
%     elseif strcmp(CopulaSpec.corrspec, 'ADCC')
%         startinvals = repmat([.01; .001; .97],ParSize,1);
%         A=[];
%         b=[];
%         nonlincon = 'Nonlincon_vine';
%     elseif strcmp(CopulaSpec.corrspec,'static')
%         startinvals = repmat(2,ParSize,1);
%         A=[];
%         b=[];
%         nonlincon = [];
%     elseif strcmp(CopulaSpec.corrspec,'Patton')
%         %         Form der Startinvals: [omega, alpha; beta]
%         startinvals = repmat([.5; .1; .1],ParSize,1);
%         %         constraint AR-Term < 1; A*x<=b
%         A = zeros(ParSize,size(startinvals,1));
%         g=3;
%         %         setze 1 an die Stelle des AR-Terms
%         for j=1:ParSize
%             A(j,g)=1;
%             g=g+3;
%         end
%         b = ones(ParSize,1)-1e-6;
%         nonlincon = [];
%     elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
%         startinvals = repmat([.05; .9],ParSize,1);
%         %         kreiere spezielle Matrix A, so das A*x<b
%         A = zeros(ParSize*2,ParSize*2);
%         k=1;
%         for i = 1:2:ParSize*2
%             A(i,k:k+1)=[1 1];
%             k=k+2;
%         end
%         b = repmat([1-(1e-10); 0],ParSize,1);
%         nonlincon = [];
%     end
% else
%     startinvals=inputstartinvals_grm(CopulaSpec,defvals);
%     A=[];
%     b=[];
%     nonlincon = [];
% end
