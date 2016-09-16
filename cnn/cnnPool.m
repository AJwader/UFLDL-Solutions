function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
PoolOut=convolvedDim / poolDim;
pooledFeatures = zeros(PoolOut, ...
        PoolOut, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
d_term= poolDim^2;
kern_conwith=ones(poolDim)/d_term;   %WE USE MEAN POOLING HERE.


for imageNum=1:numImages
    for filterNum = 1:numFilters
       
        
     %convolution with current feature map(is input).
     
     current_input = squeeze(convolvedFeatures(:,:,filterNum,imageNum));
     con_result=conv2(current_input,kern_conwith,'valid');  %This is (convdim-pollDim+1) matrix.
        
     %downsampling to get needed values for pooling layer.
     Temp=downsample(con_result,poolDim);   %This gives column wise(treat each coloum as input for downsample)  
     Temp2=downsample(Temp',poolDim);   %This gives downsample result.   
     
     Temp2=Temp2';   %This is our final  needed pooled values.  
        
        %store to pooledFeatures
     pooledFeatures(:,:,filterNum,imageNum) = Temp2;   
       
        
    end
          
end


end

