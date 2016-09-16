function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer(after pooling applied.)
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor(matrix) for storing activations
activations=cnnConvolve(filterDim, numFilters, images, Wc, bc);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled =cnnPool(poolDim, activations);



% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.


% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);
    


%probability matrix for colum wise batch image example.
probs=bsxfun(@rdivide, exp(bsxfun(@minus, bsxfun(@plus,Wd*activationsPooled,bd), max(bsxfun(@plus,Wd*activationsPooled,bd)))), sum(exp(bsxfun(@minus, bsxfun(@plus,Wd*activationsPooled,bd), max(bsxfun(@plus,Wd*activationsPooled,bd))))));
             
     
     
%%% YOUR CODE HERE %%%

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

 % save objective into cost

groundTruth=full(sparse(labels,1:numImages,1));
forlg = groundTruth.*probs;    
lgvec=log(forlg(forlg~=0));    %extract non zero of forlg and take log and store in column vector lgvec.

cost=-sum(lgvec)/numImages;  
clear lgvec forlg;
%%% YOUR CODE HERE %%%

% Makes predictions given probs.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;

% [~,preds] = max(finalout,[],1);   %finalout is final layer output.
% preds=preds';
% grad = 0;
    return;
      
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
              %error for output layer 
                   
  ErrVecforIndiv=(-1/numImages) * (groundTruth-probs); %DIVIDE BY NUMEXAMPLE(BATCHSIZE FOR MINIBATCH AND ONE IF STOCHASTIC)     
                             
              termforupsample=Wd'*ErrVecforIndiv;
              reshapTerm=reshape(termforupsample,outputDim,outputDim,numFilters,numImages);
     
              Errorupsample=zeros(convDim,convDim,numFilters,numImages);
              
              %Gradient for layer which is pooled activation reshape layer
              %that connected densely to output softmax layer.
              Wd_grad=ErrVecforIndiv*activationsPooled';
              bd_grad=sum(ErrVecforIndiv,2);
              clear ErrVecforIndiv;
              clear activationsPooled;

for imageNum = 1:numImages
  im = squeeze(images(:,:,imageNum));
  for filterNum = 1:numFilters
   
      upsampleResult=(1/(poolDim^2)).*kron(squeeze(reshapTerm(:,:,filterNum,imageNum)),ones(poolDim));
      
      Errorupsample(:,:,filterNum,imageNum)=(upsampleResult).*(activations(:,:,filterNum,imageNum).*(1-activations(:,:,filterNum,imageNum)));
      
       
      
      %Gradient for convolutional layer related parameters.
      FMerror=squeeze(Errorupsample(:,:,filterNum,imageNum));
      Fconvo=conv2(im,rot90(squeeze(FMerror),2),'valid');
      
      Wc_grad(:,:,filterNum)=squeeze(Wc_grad(:,:,filterNum)) + Fconvo;
      bc_grad(filterNum)= bc_grad(filterNum)+ sum(FMerror(:));
      
  end
  
end
              
clear activations;
clear reshapTerm;
clear Errorupsample;              
              
       


%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];



end
