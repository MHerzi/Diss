function out = NewsImpactGARCH2(startval,dist,endval,GARCHOutput,plotfigure)

% News Impact curves for univariate GARCH models (GARCH, TGARCH, GJRGARCH,
%    EGARCH); uses simulated innovations to compare different GARCH models
% 
% USAGE:
%       out = NewsImpactGARCH(startval,dist,endval,GARCHOutput)
% 
% INPUTS:
%         startval:    number where simulated Innovations should start
%             dist:    distance bewteen simulated Innovations
%           endval:    last value of simulated Innovations
%       plotfigure:    if 'on' plot news impact curves
% 
% OUTPUTS: 
%         t x k array of simulated news impact curves
%        
% References: Zivot, E., Wang, J.(2006), Modeling Financial Timer Series
% with S-PLUS"
% 
% Author: Martin Grziska, 05/15/2010

cd G:\Daten\EM_final
data=xlsread('Indices_Stocks_G6+Canada');
data=price2ret(data);
SP500=data(:,end);

GARCH{1}='GARCH';
GARCH{2}='EGARCH';
GARCH{3}='TGARCH';
GARCH{4}='GJRGARCH';

% alle Modelle GARCH(1,1)
p=ones(4,1); %archP
o=[0 ones(1,3)]; %GARCH hat keinen Asymmetrieparameter
q=ones(4,1); %garchQ


for i=1:4
    [parameters{i}, likelihood{i}, stderrors{i}, robustSE{i}, ht{i}, scores{i}, resid{i}]=ar_multigarch_grm(SP500,p(i),o(i),q(i),GARCH{i},'NORMAL',1,1);
end

for i=1:4
    if strcmp(GARCH{i},'GARCH')
        lagresid=mlag(resid{i},1);
        lagresid=lagresid(2:end);
        longh=sqrt(var(SP500));
        A = parameters{i}(1) + parameters{i}(3)*longh;
        ht = A + parameters{i}(1)*lagresid.^2;
        plotdata=[ht lagresid];
        plotdataGARCH=sortrows(plotdata,2);        
    elseif strcmp(GARCH{i},'EGARCH')
        lagresid = mlag(resid{i},1);
        lagresid=lagresid(2:end);%NewsImpact curves werden  mit gelaggten Residuen geschätzt
        longh = exp((parameters{i}(1) + parameters{i}(2)*sqrt(2/pi))/(1-parameters{i}(4))); % long run Variance
        A = longh^(2*parameters{i}(4))*exp(parameters{i}(1));
        ht = A*exp(parameters{i}(2)*(abs(lagresid)+parameters{i}(3)*lagresid)/longh); %variance
         plotdata=[ht lagresid];
        plotdataEGARCH=sortrows(plotdata,2);   
    elseif strcmp(GARCH{i},'TGARCH')
        lagresid = mlag(resid{i},1);
        lagresid=lagresid(2:end);%NewsImpact curves werden  mit gelaggten Residuen geschätzt
        
    elseif strcmp(GARCH{i},'GJRGACRH')
        lagresid = mlag(resid{i},1);
        lagresid=lagresid(2:end);%NewsImpact curves werden  mit gelaggten Residuen geschätzt
        S = (lagresid<0);
        longh = parameters{i}(1)/(1-(parameters{i}(1)+parameters{i}(3)/2)-parameters{i}(4));
        A = parameters{i}(1) + parameters{i}(4)*longh;
        ht = A + (parameters{i}(2)+parameters{i}(3)*S).*lagresid.^2;
        plotdata=[ht lagresid];
        plotdataGJRGARCH=sortrows(plotdata,2);   
    end
        


