function [ sx ] = sparserate( w)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
w1 = w(1:end-1);
sum = 0;
for i =1:length(w1)
if(w(i)==0)
    sum = sum +1;
end
end
sx = (length(w1)-sum)/(length(w1));

end

