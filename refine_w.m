function [w_tilde]=refine_w(g_bar,w_bar,L,lambda)
%This function is mainly to refine the answer w by solving the following optimization problem:
% w_tilde = arg min_{w}F(w)={<g_bar,w>+L/2*||w-w_bar||^2+\lambda*||w||_1}

V1 = g_bar-L*w_bar;
V2 = L/2.0;

w_tilde = -(V1-lambda*sign(V1))/(2*V2);
index = abs(V1)>lambda;
w_tilde = w_tilde.*index;
w_tilde(end) = -V1(end)/(2*V2);
end
