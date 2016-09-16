% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 3e-3;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
%% Gradient checking
DEBUG=false;              %set true when debgging

if DEBUG
    %here reduce computation time set small number of needed parameters.
    ei.input_dim = 200;    %here you can set input layer unit number or input dimention.
    ei.layer_sizes = [5,5,2,5,5, ei.output_dim];  %here you can set number of hidden unit and hidden layer number.
    stack = initialize_weights(ei);
    params = stack2params(stack);
    
    
   data_train=data_train(1:ei.input_dim,1:100);   %portion of data for gradient check you don't need all example for this checking process
   labels_train=labels_train(1:100);          %related lables example number.
   
   
   
   
   %get gradient by our cost function using backprop
   [cost,grad]=supervised_dnn_cost(params,ei,data_train,labels_train);
   
   %get numarical gradient by ComputerNumaricalGrad function
   numGrad= ComputeNumaricalGrad(@(x) supervised_dnn_cost(x,ei,data_train,labels_train) , params);  
   disp([numGrad grad]); 
            
   % Compare numerically computed gradients with those computed analytically
   diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

    % When your gradients are correct, congratulations!
end 
    
  
   
   


%% run training
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);





%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [],true);
[~,pred] = max(pred);

acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', 100*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', 100*acc_train);
