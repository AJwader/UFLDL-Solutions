%% We will use minFunc for this exercise, but you can use your
% own optimizer of choice
clear all;
addpath(genpath('../common/')) % path to minfunc
%% These parameters should give you sane results. We recommend experimenting
% with these values after you have a working solution.
global params;
params.m=50000; % num patches
params.patchWidth=9; % width of a patch
params.n=params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005; % sparsity cost
params.numFeatures = 50; % number of filter banks to learn
params.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity

% Load MNIST data set
data = loadMNISTImages('../common/train-images-idx3-ubyte');

%% Preprocessing
% Our strategy is as follows:
% 1) Sample random patches in the images
% 2) Apply standard ZCA transformation to the data
% 3) Normalize each patch to be between 0 and 1 with l2 normalization

% Step 1) Sample patches
patches = samplePatches(data,params.patchWidth,params.m);
% Step 2) Apply ZCA
patches = zca2(patches);
% Step 3) Normalize each patch. Each patch should be normalized as
% x / ||x||_2 where x is the vector representation of the patch
m = sqrt(sum(patches.^2) + (params.epsilon));
x = bsxfunwrap(@rdivide,patches,m);

%% Run the optimization
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 1000;
options.DERIVATIVECHECK='true';
%options.display = 'off';
options.outputFcn = @showBases;

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);






%% Gradient checking
DEBUG=false;              %set true when debgging

if DEBUG
    
 
params.m=20; % num patches
params.patchWidth=9; % width of a patch
params.n=params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005; % sparsity cost
params.numFeatures = 20; % number of filter banks to learn
params.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity


% Sample patches
patches = samplePatches(data,params.patchWidth,params.m);
% Apply ZCA
patches = zca2(patches);
% Normalize each patch. Each patch should be normalized as
% x / ||x||_2 where x is the vector representation of the patch
m = sqrt(sum(patches.^2) + (params.epsilon));
x = bsxfunwrap(@rdivide,patches,m);

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);
   
   
   
   
   %get gradient by our cost function
  [~,grad] = softICACost(randTheta, x, params);
   
   %get numarical gradient by ComputerNumaricalGrad function
% avggrad=grad_check(@softICACost,randTheta,100,x,params);  %using  grad_check.
   numGrad= ComputeNumaricalGrad(@(theta) softICACost(theta,x,params) ,randTheta );  
   disp([numGrad grad]); 
            
   % Compare numerically computed gradients with those computed analytically
   diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

    % When your gradients are correct, congratulations!
end





% optimize
% [opttheta, cost, exitflag] = minFunc( @(weights) softICACost(weights, x, params), randTheta, options); % Use x or xw
[opttheta, cost, exitflag] = minFunc(@softICACost, randTheta, options,x,params); % Use x or xw
% display result
W = reshape(opttheta, params.numFeatures, params.n);
display_network(W');
