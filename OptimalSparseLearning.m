function [ w_tilde,t_Optimal ] = OptimalSparseLearning(param)
%SPARSELEARNINGONAVERAGEGRADIENT Summary of this function goes here
%   Detailed explanation goes here
%f(w)=(1.0/2n)*\Sigma{(y_i-x_i*w)^2}+(\rho/2)*(||w||_2)^2+\lambda*||w||_1
domainsize = param.domainsize;
lambda = param.lambda;
rho = param.rho;
alpha = param.alpha;
truerho = param.truerho;
if(param.iternum==-1)
    T =length(param.Y_train);
else
    T = param.iternum;
end
T_1 = floor(T*(1-alpha));
T_2 = T-T_1;
%alpha_SGD = 0.01;
L = param.L;

param_alphaSGD.lambda = lambda;
param_alphaSGD.rho = rho;
param_alphaSGD.iternum = T_1;
param_alphaSGD.alpha_SGD = param.alpha_SGD;
%param_alphaSGD.functionname = param.functionname;
param_alphaSGD.truerho = truerho;
param_alphaSGD.X_train = param.X_train;
param_alphaSGD.Y_train = param.Y_train;
param_alphaSGD.domainsize = param.domainsize;
tic;
[w_bar1] = alphaSGD(param_alphaSGD);         %w_bar1 is \bar{w}_{1-\alpha}

sum_g_bar1 =0;
for i = T_1+1:T_1+T_2
    x_i = param.X_train(:,i);
    y_i =param.Y_train(i);
    x_len = [x_i;1];
    wx = -y_i*(w_bar1'*x_len);
    if(wx<10)
        gh_t = -exp(wx)/(1+exp(wx))*y_i*x_len+rho*w_bar1;
    else
        gh_t = -1.0/(1+exp(-wx))*y_i*x_len+rho*w_bar1;
    end
    sum_g_bar1 = sum_g_bar1 +gh_t;
end
g_bartemp = sum_g_bar1*(1.0/T_2);
V1 = g_bartemp-L*w_bar1;
V2 = L/2.0;

w_tilde = -(V1-lambda*sign(V1))/(2*V2);
index = abs(V1)>lambda;
w_tilde = w_tilde.*index;
w_tilde(end) = -V1(end)/(2*V2);
t_Optimal = toc;
end

