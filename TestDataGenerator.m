function [ X,Y] = TestDataGenerator( param_test )
%TESTDATAGENERATOR Summary of this function goes here
%   Detailed explanation goes here

datasetname = param_test.datasetname_test;
if(strcmp(datasetname,'MNIST')==1) %for multi-class data sets
    v1 = param_test.v1;
    v2 = param_test.v2;
    
    
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
    
    images = loadMNISTImages('DataSet/Test/MNIST/t10k-images.idx3-ubyte');
    labels = loadMNISTLabels('DataSet/Test/MNIST/t10k-labels.idx1-ubyte');
    
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
    
    X = X';
    Y = Y';
    
    
    
    
else % for two class data sets
    datasetname_here = ['DataSet/Test/',datasetname,'/',datasetname];
    [Y,X] = libsvmread(datasetname_here);
    %%%%%%begin normalize%%%%%%%%%
    %min_inte = ones(size(X))*diag(min(X));
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
    %X = sparse(X);
    %%%%%%end normalize%%%%%%%
    
    max_Y = max(Y);
    min_Y = min(Y);
    Y = (Y-(max_Y+min_Y)/2)/((max_Y-min_Y)/2);
end




end

