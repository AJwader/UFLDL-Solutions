%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);  

    %========= RICA:
% %%% YOUR CODE HERE %%%
costTerm1=sqrt(((W*x).^2) + params.epsilon);
costTerm2=(W'*W*x - x).^2;
% 

% % 
%  cost=(sum(costTerm1(:)) + sum(costTerm2(:)));
cost=sum(costTerm1(:)) + sum(costTerm2(:)); 
cost=cost/params.m;

                             
                                        
 





%% Comuting Gradient


gradient1=2*((W'*W*x - x) * (W*x)');  %gradient w.r.t W transpose.
gradient2=W*(2*(W'*W*x - x))*x';      %gradient w.r.t W.
gradpart=((W*x)./costTerm1)*x';       
Wgrad=(gradient2 + gradient1') + gradpart;           %final gradient w.r.t W is sum of grad2 and grad1 transpose.
Wgrad=Wgrad/params.m;

% % unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);



end


