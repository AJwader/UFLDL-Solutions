function [ numgrad ] = ComputeNumaricalGrad(J , theta )
% compute numarical gradient for checking with our backprop gradient
% take input as cost function and parameter of network as theta

numgrad = zeros(size(theta));


epsilon=1e-4;
n=size(numgrad,1);
%I=eye(n,n);



 %% using loop

% for i = 1:size(numgrad)
%     i
%    eps_vec = I(:,i) * eps;
%    numgrad(i) = (J(theta + eps_vec) - J(theta - eps_vec)) ./ (2 * eps);
%     
% end

J1 = zeros(1, 1);
J2 = zeros(1, 1);
temp1 = zeros(size(theta));
temp2 = zeros(size(theta));

for i = 1 : n
   i
   temp1 = theta;
   temp2 = theta;
   temp1(i) = temp1(i) + epsilon;
    temp2(i) = temp2(i) - epsilon;
    J1  = J(temp1);
    J2  = J(temp2);
    numgrad(i) = (J1 - J2) / (2*epsilon);
end



end
     