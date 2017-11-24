function [ v_loss ] = calLoss(w,param )
%CALLOSS Summary of this function goes here
rho = param.rho;
lambda = param.lambda;
w1 = w(1:end-1);
b = w(end);
result_temp = -(param.X_train'*w1+b).*param.Y_train;
sum1 = log(1+exp(result_temp));
sum1 = sum(sum1);
v_loss = sum1/length(param.Y_train)+rho/2.0*(w'*w) +lambda*norm(w1,1);
end




