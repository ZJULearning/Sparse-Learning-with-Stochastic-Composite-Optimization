function [ tsx ] = truncatedsparserate( w,mythreshold )
%TRUNCATEDSPARSERATE Summary of this function goes here
%   Detailed explanation goes here
epsilon =mythreshold;
sum = 0;
w1 = w(1:end-1);
for i = 1:length(w1)
if(abs(w1(i))<=epsilon)
    sum = sum +1;
end
end
tsx = (length(w1)-sum)/length(w1);

end

