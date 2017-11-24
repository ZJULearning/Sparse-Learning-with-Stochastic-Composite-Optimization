function [X1,Y1] = datagenerator( param_data )
%DATAGENERATOR Summary of this function goes here
%   Detailed explanation goes here
datasetname = param_data.datasetname;


if(strcmp(datasetname,'MNIST')==1)
    v1 = param_data.v1;
    v2 = param_data.v2;
    
    
    if(v1==v2)
        fprintf('v1 and v2 should be different numbers');
        exit(0);
    end
    
    if(v1<v2)
        mymin = v1;
        mymax = v2;
    else
        mymin = v2;
        mymax = v1;
    end
    
    images = loadMNISTImages('DataSet/Train/MNIST/train-images.idx3-ubyte');
    labels = loadMNISTLabels('DataSet/Train/MNIST/train-labels.idx1-ubyte');
    
    idx = 1;
    for i = 1:length(labels)
        if(labels(i)==mymin||labels(i)==mymax)
            X(:,idx) = images(:,i);
            if(labels(i)==mymin)
                Y(idx) = -1;
            else
                Y(idx) = 1;
            end
            idx = idx +1;
        end
    end
    [ignore,p] = sort(rand(1,length(Y)));
    out=p';
    
    for i = 1:length(Y)
        X1(:,i) = X(:,out(i));
        Y1(i) = Y(out(i));
    end
    Y1 = Y1';
    
    
    
else
    datasetname_here = ['DataSet/Train/',datasetname,'/',datasetname];
    [Y,X] = libsvmread(datasetname_here);
%     if(length(Y)>100000)
%         Y = Y(1:100000);
%         X = X(1:100000,:);
%     end
    
    %%%%%%begin normalize%%%%%%%%%
    gap_vector = max(X)-min(X);
    gap_inte = 0.*speye(size(diag(gap_vector)));
    for i = 1:length(gap_vector)
        if(gap_vector(i)~=0)
            gap_inte(i,i)= 1.0/gap_vector(i);
        end
    end
    if(strcmp(datasetname,'realsim')==1||strcmp(datasetname,'rcv1_test.binary')==1||strcmp(datasetname,'rcv1_train.binary')==1||strcmp(datasetname,'epsilon_normalized.t')==1)
        min_X = min(X);
        
        
        for i = 1:length(min_X)
            X(:,i)=X(:,i)-min_X(i);
        end
    else
        min_inte = ones(size(X))*diag(min(X));
        X = X - min_inte;
    end
    X = X*gap_inte;
    max_Y = max(Y);
    min_Y = min(Y);
    Y = (Y-(max_Y+min_Y)/2)/((max_Y-min_Y)/2);
    %X = sparse(X);
    %%%%%%end normalize%%%%%%%
    
    positive_rate = (length(Y)+sum(Y))/(2*length(Y))
    
    
    
    [ignore,p] = sort(rand(1,length(Y)));
    q = [1:length(Y)];
    A = 0.*speye(length(Y),length(Y));
    idx = (p-1)*length(Y)+q;
    A(idx)=1;
    X = A*X;
    Y1 = A*Y;
    
    X1 = X';
end





end

