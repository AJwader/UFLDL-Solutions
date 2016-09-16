addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (0 to 9).

binary_digits = false;
num_classes = 10;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];
train.y = train.y+1; % make labels 1-based. for computation easy in matlab.
test.y = test.y+1; % make labels 1-based.

        
% Training set info
m=size(train.X,2);
n=size(train.X,1);

% Initialize theta.  We use a matrix where each column corresponds to a class,
% and each row is a classifier coefficient for that class.
% Inside minFunc, theta will be stretched out into a long vector (theta(:)).
% We only use num_classes-1 columns, since the last column is always assumed 0.
theta = rand(n,num_classes)*0.001;



% For Gradient checking use less parameter for fast computation.
DEBUG=false;

if DEBUG
  n=8;
  train.X=randn(n,100);
  train.y=randi(10,1,100);    
  
  %here we use theta for n by 10 that fit computation with groundtruth.
  theta = rand(n,num_classes)*0.001;
  
  
    theta=theta(:);  %make vector for ComputeNumaricalGrad function.
    
    [~,grad]=softmax_regression_vec(theta,train.X,train.y);
    %we can also use grad_check for gradient checking.
%     avg_error=grad_check(@softmax_regression_vec,theta,100,train.X,train.y);
    numGrad = ComputeNumaricalGrad( @(x) softmax_regression_vec(x,train.X,train.y), theta);

    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 

    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

    % When your gradients are correct, congratulations!
end




%Implement softmax_regression_vec

% Train softmax classifier using minFunc
options = struct('MaxIter', 700);



% Call minFunc with the softmax_regression_vec.m file as objective.
%
% TODO:  Implement batch softmax regression in the softmax_regression_vec.m
% file using a vectorized implementation.
%
tic;
theta(:)=minFunc(@softmax_regression_vec, theta(:), options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);


%theta=[theta, zeros(n,1)]; % expand theta to include the last class.

% Print out training accuracy.
tic;
accuracy = multi_classifier_accuracy(theta,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);

% Print out test accuracy.
accuracy = multi_classifier_accuracy(theta,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);


% % for learning curves
% global test
% global train
% test.err{end+1} = multi_classifier_accuracy(theta,test.X,test.y);
% train.err{end+1} = multi_classifier_accuracy(theta,train.X,train.y);
