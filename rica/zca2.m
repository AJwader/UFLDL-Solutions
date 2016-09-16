function [Z,V] = zca2(x)
%  epsilon=0;   %when using no regularization.
 epsilon = 1e-4;   % epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.
%%================================================================
%% Step 0b: Zero-mean the data (by row)
mn_all=mean(x,2);      
x=bsxfun(@minus,x,mn_all);

 sigma=x*x'/size(x,2); 
 [U S, V]=svd(sigma);

  %% Step 4a: Implement PCA with whitening and regularisation
%  Xrot=U'*x;
%  diagS=diag(S);
% XpcaWhite = bsxfun(@rdivide,Xrot,sqrt(diagS+epsilon));   %can also use this
XpcaWhite=diag(1./sqrt(diag(S) + epsilon)) * U' * x;
 

 %% Step 5: Implement ZCA whitening
xZCAWhite = U * XpcaWhite;
Z=xZCAWhite;


% Visualise the data, and compare it to the raw data.
randsel = randi(size(x,2),200,1);    %for selecting random selection from data
figure('name','ZCA whitened images');
display_network(Z(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));
