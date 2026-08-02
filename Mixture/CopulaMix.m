
function [LL_mix] = CopulaMix(weight,data)


[Output1 LL1 exitflag1 numParam1] = copulafitmultivariat('clayton', data);
[Output2 LL2 exitflag2 numParam2] = copulafitmultivariat('gumbel', data);

LL_mix = weight(1)*LL2+weight(2)*LL4;