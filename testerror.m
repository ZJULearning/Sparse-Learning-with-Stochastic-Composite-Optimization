function [ errorrate,t_use] = testerror( w,param_data )
w1 = w(1:end-1);
b = w(end);

tic;
result_temp = (param_data.X_test*w1+b).*param_data.Y_test;
result_temp = (result_temp<0);
sum_here = sum(result_temp);
t_use = toc;
errorrate = sum_here /length(param_data.Y_test);

end

