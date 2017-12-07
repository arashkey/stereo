 
nvars=8;
lb=[1 1 1 1 1 1 1 1 ];
ub=[32 32 32 32 32 32 32];
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = optimoptions('ga');
%% Modify options setting
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'PlotFcn', { @gaplotbestf });
[x,fval,exitflag,output,population,score] = ga(@fitfunc,nvars,[],[],[],[],lb,ub,[],[],options)
disp([x,fval,exitflag,output,population,score])
