function [ w_tilde,t_Optimal ] = testforOptimalLearning(param_here )
%TEST2 Summary of this function goes here
%   Detailed explanation goes here
param.lambda = param_here.lambda;
param.rho = param_here.rho;

param.alpha = param_here.alpha;
param.alpha_SGD = param_here.alpha_SGD;
param.iternum = param_here.iternum;
param.L=param_here.L;
param.truerho = param_here.truerho;
param.X_train = param_here.X_train;
param.Y_train = param_here.Y_train;
param.domainsize = param_here.domainsize;

[w_tilde,t_Optimal] = OptimalSparseLearning(param);


end

