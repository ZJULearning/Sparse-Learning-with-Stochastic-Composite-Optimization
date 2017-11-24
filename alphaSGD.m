function [ w,t_alphaSGD ] = alphaSGD(param )
%The implementation of the icml paper "Making Gradient Descent Optimal for Strongly Convex Stochastic Optimization"
%\phi(w) = log(1+exp(-y(w1x+b)))+\rho/2 ||w||^2 + \lambda ||w1||_1

domainsize = param.domainsize;
lambda = param.lambda;
rho = param.rho;
truerho = param.truerho;
if(param.iternum==-1)
    T=length(param.Y_train);
else
    T = param.iternum;
end
alpha = param.alpha_SGD;
T_begin = floor(T*(1-alpha))+1;

if(T>length(param.Y_train))
    fprintf('The iteration times should be smaller than the size of training data!');
    exit(0);
end
tic;
w_t = randn(size(param.X_train,1)+1,1);
w_sum = 0;

for t = 1:T
eta_t = 1.0/(truerho*t);
%x_t = (X(t,:))';
x_t = param.X_train(:,t);
y_t = param.Y_train(t);
signwt = sign(w_t);
signwt(end) = 0;
%gh_t = subgradientcalculator(x_t,y_t,w_t,param) + lambda*signwt;
x_len = [x_t;1];

wx = -y_t*(w_t'*x_len);
if(wx<10)
gh_t = -exp(wx)/(1+exp(wx))*y_t*x_len+rho*w_t+ lambda*signwt;
else
gh_t = -1.0/(1+exp(-wx))*y_t*x_len+rho*w_t+ lambda*signwt;
end
%gh_t = -exp(-y_t*(w_t'*x_len))/(1+exp(-y_t*(w_t'*x_len)))*y_t*x_len+rho*w_t+ lambda*signwt;

w_t = w_t -eta_t*gh_t;
    %try
    tmp_norm = norm(w_t,2);
    if (tmp_norm>domainsize);
        w_t = w_t/tmp_norm*domainsize;
    end
    %try end

if(t>=T_begin)
    w_sum = w_sum +w_t;
end


end

w = w_sum*(1.0/(T-T_begin+1));

t_alphaSGD= toc;
end

