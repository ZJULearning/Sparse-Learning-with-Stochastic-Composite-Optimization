function [ w_tilde,t_Last ] = SparseLearningBasedontheLastSolution(param )
%This function is mainly to implement the Sparse Learning Based on the Last Solution
%   f(w)=(1.0/2n)*\Sigma{(y_i-x_i*w)^2}+(\rho/2)*(||w||_2)^2+\lambda*||w||_1
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
L = param.L;
T_begin = floor((1-alpha)*T)+1;
tic;
w_t =randn(size(param.X_train,1)+1,1);
gt_sum = 0;
for t = 1 : T
    eta_t = 1.0/(truerho*t); 
    x_t = param.X_train(:,t);
    y_t =param.Y_train(t);
    signwt = sign(w_t);
    signwt(end) = 0;
    x_len = [x_t;1];
    wx = -y_t*(w_t'*x_len);
    if(wx<10)
        g_bar_t = -exp(wx)/(1+exp(wx))*y_t*x_len+rho*w_t + lambda*signwt;
    else
        g_bar_t = -1.0/(1+exp(-wx))*y_t*x_len+rho*w_t + lambda*signwt;
    end
    if(t>=T_begin)
        x_len = [x_t;1];
        wx = -y_t*(w_t'*x_len);
        if(wx<10)
            g_here = -exp(wx)/(1+exp(wx))*y_t*x_len+rho*w_t;
        else
            g_here = -1.0/(1+exp(-wx))*y_t*x_len+rho*w_t;
        end
        gt_sum = gt_sum+g_here;    
    end
    w_t = w_t -eta_t*g_bar_t;
    
    tmp_norm = norm(w_t,2);
    if (tmp_norm>domainsize)
        w_t = w_t/tmp_norm*domainsize;
    end
end
w_T = w_t;

g_bar_alpha = gt_sum /(T-T_begin +1);
L_temp = 2*L;
V1 = g_bar_alpha-L_temp*w_T;
V2 = L_temp/2.0;

w_tilde = -(V1-lambda*sign(V1))/(2*V2);
index_here = abs(V1)>lambda;
w_tilde = w_tilde.*index_here;
w_tilde(end) = -V1(end)/(2*V2);

t_Last = toc;

end
