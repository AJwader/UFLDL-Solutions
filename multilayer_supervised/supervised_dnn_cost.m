function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%YOUR CODE HERE %%%

%Here I use FOR loop to implement forward and backward pass for given
%number of layers for example you can try different numbers of layers and
%this code will work you can set number of hidden layers,hidden units so on from ei. 


FpropResults=cell(numel(ei.layer_sizes)+1,1);  %forward prop result for all layer here first layer output is data itself.
FpropResults{1,1}.Result=data;   %storing data in cell to use in loop computation.
m=size(data,2);


Decaycost=0;



%computing forward prop for given network and weight decay.
for i=1:numel(ei.layer_sizes)
    
    if i==numel(ei.layer_sizes)          %this gives activation value for output layer using logistic function.
        
    z=bsxfun(@plus,stack{i,1}.W*FpropResults{i,1}.Result,stack{i,1}.b);
    a=act(z,'logistic'); 
    FpropResults{i+1,1}.Z=z;  
    FpropResults{i+1,1}.Result=a; %final output layer activation result by sigmoid function.
     
       
    else                     %this gives activation values for hidden layers using our selected function.
    z=bsxfun(@plus,stack{i,1}.W * FpropResults{i,1}.Result, stack{i,1}.b);
    a=act(z,ei.activation_fun);
    FpropResults{i+1,1}.Z=z;   %here z is input to units in layer. note that input for first layer is data itself.
    FpropResults{i+1,1}.Result=a;  

    end
    
    %Computing wight decay for cost
    Decaycost=  Decaycost + sum(sum(stack{i,1}.W.^2)); 

end


WeightDecayCost=(ei.lambda/2)*Decaycost;  %weight decay for cost

clear z a i;   


%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  pred_prob=FpropResults{numel(ei.layer_sizes)+1,1}.Result;  %give final output layer result.which is probability for classes.
  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%


groundTruth = full(sparse(labels, 1:m, 1)); %10 by m matrix.


p=bsxfun(@rdivide,exp(FpropResults{numel(ei.layer_sizes)+1,1}.Z),sum(exp(FpropResults{numel(ei.layer_sizes)+1,1}.Z)));  %probability matrix for all m examples.

cost=(-sum(sum(groundTruth.*log(p))))  + (WeightDecayCost);



%% compute gradients using backpropagation

%%% YOUR CODE HERE %%%
 
ErrorStore=cell(numHidden+1,1);    %store computed error for layers given by backprop first cell store output layer error and so on.

%error for output layer, this is useful for hidden layer error computation in loop

finalErr=groundTruth-p;
 
ErrVecforIndiv=(-1) * finalErr;     % 10 by m matrix for output layer error term for all coloum vise example


ErrorStore{1,1}.Error=ErrVecforIndiv; %Error needed for gradient computation this store error from output layer to first hidden layer order this is useful for loop computation.
stackforBP=stack(2:end);  
stackforBP=flip(stackforBP);       %Flip needed wights for gradient Computataion.
FpropResultforBP=cell(numHidden,1);
FpropResultforBP=FpropResults(2:end-1);
FpropResultforBP=flip(FpropResultforBP);  %Flip needed activation result for gradient.


%compute error for hidden layers.
for j=1:numHidden
    
    %derrforZ=(FpropResultforBP{j,1}.Result).*(1-FpropResultforBP{j,1}.Result);
    derrforZ=dact(FpropResultforBP{j,1}.Z,ei.activation_fun);
    LError=(stackforBP{j,1}.W'*ErrorStore{j,1}.Error).*(derrforZ);
    
    ErrorStore{j+1,1}.Error=LError;    %storing computed error to errorstore for next computation.
   
    
end

clear derrforZ LError j;  

%% Computing desire partial derivatives

ErrorStore=flip(ErrorStore);   %flip given error cell for derivative computation.

%Partial derivatives from layer one to  last hidden layer.
for k=1:numel(ei.layer_sizes)
   
    WDerrLayer=(ErrorStore{k,1}.Error) * (FpropResults{k,1}.Result')+(ei.lambda*stack{k,1}.W);
    bDerrLayer=sum(ErrorStore{k,1}.Error,2);
    
    
    %Storing gradient to gradStack.
    gradStack{k,1}.W=WDerrLayer;  
    gradStack{k,1}.b=bDerrLayer;
    
    
end

clear WDerrLayer bDerrLayer k;  


%% reshape gradients into vector
[grad] = stack2params(gradStack);




end
% Activation functions
function f=act(z,type)
       
       switch type
    case 'logistic'
        f=1./(1+exp(-z));
    case 'tanh'
        f=tanh(z);
    case 'relu'
        f=max(0,z);
       end   

end

%Derivative for activation function.
function df=dact(z,type)

      switch type
         
          case 'logistic'
              df=act(z,type).*(1-act(z,type));
          case 'tanh'
              df=1-act(z,type).^2;
          case 'relu'     
              df=double(z>0);  
          
      end

end









