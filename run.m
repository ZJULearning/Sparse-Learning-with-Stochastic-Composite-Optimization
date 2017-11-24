function [ output_args ] = run( )
%This is the implementation of our work "Sparse Learning with Stochastic Composite Optimization"
%In this function, we run every algorithm for many(outnum) times, and report the 
%averaged results (like Objective value over the iterations, time cost, sparsity, and so on).
% For any problem, please contact Weizhong Zhang(zhangweizhongzju@gmail.com) 

datasetname = 'MNIST';
datasetname_test = 'MNIST';
v1= 2; v2=3;%% use digits 2 and 3 as the training data
rho = 0.01;
lambda = 0.01;
outnum = 100;


maxiter=10000;
domainsize = 50;
mythreshold = 10^(-6);
myalpha = 0.3;
alpha_SGD = 0.3;
param.rho = rho;
param.lambda = lambda;
param.domainsize = domainsize;
param_data.datasetname=datasetname;
param_data.datasetname_test = datasetname_test;
param_data.v1 = v1;
param_data.v2 = v2;
param.L =10;
param.truerho = rho;
param.alpha_SGD= alpha_SGD;
param.alpha = myalpha;
gap1 = 1500;
gap2 = 8000;
Obj_Optimal = 0;
Obj_Last = 0;
Obj_Average = 0;
dx_Optimal=0;
tdx_Optimal=0;
dx_Last=0;
tdx_Last=0;
dx_Average = 0;
tdx_Average = 0;

error_Last =0;
t_Last =0;
error_Optimal=0;
t_Optimal =0;
error_Average = 0;
t_Average = 0;


t_error_Optimal = 0;
t_time_Optimal = 0;
t_error_Last = 0;
t_time_Last = 0;
t_error_Average = 0;
t_time_Average = 0;

t_cost_Optimal_sum =0;
t_cost_Last_sum = 0;
t_cost_Average_sum =0;

[X_test,Y_test]=TestDataGenerator(param_data);
param.X_test = X_test;
param.Y_test = Y_test;

for outer = 1:outnum
    fprintf('The %d-th runing of the algorithms....\n', outer);
    [X_train,Y_train] = datagenerator(param_data);
    param.X_train=X_train;
    param.Y_train=Y_train;
    iternum =0;
    flag = 0;
    idx =1;
    for i = 1:1000
        if(iternum<=2500)
            iternum = iternum +100;
        elseif(iternum>2500&&iternum<=10000)
            iternum = iternum +gap1;
        else
            iternum = iternum +gap2;
        end
        if(iternum>maxiter&&flag==0)
            iternum =maxiter;
        end
        if(iternum<=maxiter&&flag==0)
            param.iternum = iternum;
            [w_optimal,t_Optimal_train] = testforOptimalLearning(param );
            ObjX_optimal(idx)=iternum ;
            ObjY_optimal(idx)=calLoss(w_optimal,param);
            dx_optimal_temp(idx) = sparserate(w_optimal);
            tdx_optimal_temp(idx) = truncatedsparserate(w_optimal,mythreshold);
            [a,b] = testerror(w_optimal,param);
            error_Optimal_temp(idx) = a;
            t_Optimal_temp(idx) =b;
            [t_error,t_timeuse] = truncated_testerror( w_optimal,mythreshold, param );
            t_error_Optimal_temp(idx) = t_error;
            t_time_Optimal_temp(idx) =t_timeuse;
            t_cost_Optimal(idx)= t_Optimal_train;
            
            
            [w_last,t_Last_train] = testforSparseLearningBasedontheLastSolution(param );
            ObjX_last(idx)=iternum ;
            ObjY_last(idx)=calLoss(w_last,param);
            dx_last_temp(idx) = sparserate(w_last);
            tdx_last_temp(idx) = truncatedsparserate(w_last,mythreshold);
            [a,b] = testerror(w_last,param);
            error_Last_temp(idx) = a;
            t_Last_temp(idx) =b;
            [t_error,t_timeuse] = truncated_testerror( w_last,mythreshold, param );
            t_error_Last_temp(idx) = t_error;
            t_time_Last_temp(idx) =t_timeuse;
            t_cost_Last(idx) = t_Last_train;
            
            
            [w_average,t_Average_train] = testforSparseLearningBasedontheAverageSolution(param);
            ObjX_average(idx)=iternum ;
            ObjY_average(idx)=calLoss(w_average,param);
            dx_average_temp(idx) = sparserate(w_average);
            tdx_average_temp(idx) = truncatedsparserate(w_average,mythreshold);
            [a,b] = testerror(w_average,param);
            error_Average_temp(idx) = a;
            t_Average_temp(idx) =b;
            [t_error,t_timeuse] = truncated_testerror( w_average,mythreshold, param );
            t_error_Average_temp(idx) = t_error;
            t_time_Average_temp(idx) =t_timeuse;
            t_cost_Average(idx)=t_Average_train;
            
            idx = idx+1;
        end
        if(iternum >=maxiter)
            flag = 1;
        end
        
    end
    
    Obj_Optimal = Obj_Optimal+ObjY_optimal;
    OptimalObjAll(outer,:) = ObjY_optimal;
    dx_Optimal = dx_Optimal + dx_optimal_temp;
    tdx_Optimal = tdx_Optimal + tdx_optimal_temp;
    error_Optimal = error_Optimal + error_Optimal_temp;
    t_Optimal = t_Optimal + t_Optimal_temp;
    t_error_Optimal = t_error_Optimal+t_error_Optimal_temp;
    t_time_Optimal = t_time_Optimal +t_time_Optimal_temp;
    t_cost_Optimal_sum = t_cost_Optimal_sum + t_cost_Optimal;
    
    Obj_Last = Obj_Last+ObjY_last;
    LastObjAll(outer,:) = ObjY_last;
    dx_Last = dx_Last + dx_last_temp;
    tdx_Last = tdx_Last + tdx_last_temp;
    error_Last = error_Last + error_Last_temp;
    t_Last = t_Last + t_Last_temp;
    t_error_Last = t_error_Last+t_error_Last_temp;
    t_time_Last = t_time_Last +t_time_Last_temp;
    t_cost_Last_sum = t_cost_Last_sum + t_cost_Last;
    
    Obj_Average = Obj_Average+ObjY_average;
    AverageObjAll(outer,:) = ObjY_average;
    dx_Average = dx_Average + dx_average_temp;
    tdx_Average = tdx_Average + tdx_average_temp;
    error_Average = error_Average + error_Average_temp;
    t_Average = t_Average + t_Average_temp;
    t_error_Average = t_error_Average+t_error_Average_temp;
    t_time_Average = t_time_Average +t_time_Average_temp;
    t_cost_Average_sum = t_cost_Average_sum + t_cost_Average;
    
    
end
Obj_Optimal = Obj_Optimal/outnum;
Var_Optimal = var(OptimalObjAll);
dx_Optimal = dx_Optimal/outnum;
tdx_Optimal = tdx_Optimal/outnum;
error_Optimal = error_Optimal/outnum;
t_Optimal = t_Optimal/outnum;
t_error_Optimal = t_error_Optimal/outnum;
t_time_Optimal = t_time_Optimal/outnum;
train_time_Optimal = t_cost_Optimal_sum/outnum;

Obj_Last = Obj_Last/outnum;
Var_Last = var(LastObjAll);
dx_Last = dx_Last/outnum;
tdx_Last = tdx_Last/outnum;
error_Last = error_Last/outnum;
t_Last = t_Last/outnum;
t_error_Last = t_error_Last/outnum;
t_time_Last = t_time_Last/outnum;
train_time_Last = t_cost_Last_sum/outnum;

Obj_Average = Obj_Average/outnum;
Var_Average = var(AverageObjAll);
dx_Average = dx_Average/outnum;
tdx_Average = tdx_Average/outnum;
error_Average = error_Average/outnum;
t_Average = t_Average/outnum;
t_error_Average = t_error_Average/outnum;
t_time_Average = t_time_Average/outnum;
train_time_Average = t_cost_Average_sum/outnum;

fprintf('Saving the results....\n');
plot(ObjX_optimal,Obj_Optimal,'r-<','linewidth',1);
hold on;
plot(ObjX_last,Obj_Last,'k-^','linewidth',1);
plot(ObjX_average,Obj_Average,'m-square','linewidth',1);

legend('OptimalSL','LastSL','AverageSL');
xlabel('Iteration');
ylabel('Objective');

str1 = num2str(lambda);
str2 = num2str(rho);

if(strcmp(datasetname,'MNIST')==1)
    resultname1 =['Result/MNIST/',num2str(v1),num2str(v2),'result','rho=',str2, 'lambda=',str1];
else
    resultname1 =['Result/',datasetname,'/',datasetname,'result','rho=',str2, 'lambda=',str1];
end
resultname3 =[resultname1,'.fig'];
saveas(gca, resultname3,'fig');
resultname2 = [resultname1,'.mat'];

if(strcmp(datasetname,'MNIST')==1)
    resultname4 = ['Result/MNIST/',num2str(v1),num2str(v2),'ResultAll','rho=',str2, 'lambda=',str1,'.mat'];
else
    resultname4 = ['Result/',datasetname,'/',datasetname,'ResultAll','rho=',str2, 'lambda=',str1,'.mat'];
end
save(resultname4,'OptimalObjAll','Var_Optimal','LastObjAll','Var_Last','AverageObjAll','Var_Average');
save(resultname2,'w_optimal','ObjX_optimal','Obj_Optimal','dx_Optimal','tdx_Optimal','error_Optimal','t_Optimal','t_error_Optimal','t_time_Optimal','train_time_Optimal','w_last', 'ObjX_last','Obj_Last','dx_Last','tdx_Last','error_Last','t_Last','t_error_Last','t_time_Last','train_time_Last','w_average','ObjX_average','Obj_Average','dx_Average','tdx_Average','error_Average','t_Average','t_error_Average','t_time_Average','train_time_Average');
close;
end

