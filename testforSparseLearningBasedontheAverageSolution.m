function [ w_tilde,t_Average ] = testforSparseLearningBasedontheAverageSolution(param_here)
%TESST Summary of this function goes here
%   Detailed explanation goes here
param.domainsize = param_here.domainsize;

param.lambda = param_here.lambda;
param.rho = param_here.rho;
param.functionname = 'ridgeSquareSparse';
param.truerho = param_here.truerho;

param.alpha = param_here.alpha;
param.iternum = param_here.iternum;
param.L = param_here.L;
param.X_train = param_here.X_train;
param.Y_train = param_here.Y_train;
[w_tilde,t_Average] = SparseLearningBasedontheAverageSolution(param);

end

