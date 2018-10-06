
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Code 04 : Stacked Auto-encoder(SAE) Neural Network            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note : PLEASE run the model sequentially! Do not clear the Workspace!
% Here we feed the model with the best set of parameters (optimised) 
% which was experimented with 'Finalcode05_SAEgridsearch'

%% 1. PRE-TRAINING phase

%deep_autoencoder=init(deep_autoencoder); %for the case of retraining

hiddenSize_ae1 = 10;
autoenc_ae1 = trainAutoencoder(X_train,hiddenSize_ae1,...
    'MaxEpochs',1000,...  %Maximum number of training epochs
    'EncoderTransferFunction','logsig',... %logsig, satlin(positive saturating linear transfer func)
    'DecoderTransferFunction','logsig',... ; %logsig, satlin, purelin
    'ScaleData',true,...;rescale the input data
    ... %loss function = MSE + L2 Regularisation + Sparsity regularization
    'L2WeightRegularization',0.001,... %The coefficient for the L2 weight regularizer in the LossFunction
    'SparsityRegularization',0.98,... %The coefficient for the L2 weight regularizer in the LossFunction
    'SparsityProportion',0.09);  % A low value for SparsityProportion usually leads to each neuron in the hidden layer "specializing" 
    % by only giving a high output for a small number of training examples



%Extract the features in the hidden layer -> to feed into the second auto-encoder!
features_ae1 = encode(autoenc_ae1,X_train);

%Train the second autoencoder using the features from the first autoencoder. 

hiddenSize_ae2 = 10;
autoenc_ae2 = trainAutoencoder(features_ae1,hiddenSize_ae2,...
    'MaxEpochs',1000,...  %Maximum number of training epochs
    'EncoderTransferFunction','logsig',... %logsig, satlin(positive saturating linear transfer func)
    'DecoderTransferFunction','logsig',... ; %logsig, satlin, purelin
    'ScaleData',true,...;rescale the input data
    ... %loss function = MSE + L2 Regularisation + Sparsity regularization
    'L2WeightRegularization',0.001,... %The coefficient for the L2 weight regularizer in the LossFunction
    'SparsityRegularization',0.98,... %The coefficient for the L2 weight regularizer in the LossFunction
    'SparsityProportion',0.09);  % A low value for SparsityProportion usually leads to each neuron in the hidden layer "specializing" 
    % by only giving a high output for a small number of training examples


%Extract the features in the hidden layer -> to feed into the output layer
features_ae2 = encode(autoenc_ae2,features_ae1);

%% 2. FINE-TUNING phase
%Now we fine-tuning the model for classification task
%Train a softmax layer for classification using the features, features2, 
%from the second autoencoder, autoenc_ae2.

softnet = trainSoftmaxLayer(features_ae2,T_train,'LossFunction','mse','MaxEpochs',1000);


%Stack all auto-encoders and the softmax layer to form a deep network.
deep_autoencoder = stack(autoenc_ae1,autoenc_ae2,softnet);

%% 4. Train the deep network with training data

%feed the training data and train the classfier
deep_autoencoder.trainParam.epochs = 1000;	%set the maximum epoch
[deep_autoencoder,tr] = train(deep_autoencoder,X_train,T_train);


%Estimate the activity types using deep stacked auto-encoders
train_activity_type = deep_autoencoder(X_train);


%% 3. Test

test_activity_type = deep_autoencoder(X_test);

% Perform vec2ind()
% The network outputs will be in the range 0 to 1, so vec2ind function used
% to get the class indices as the position of the highest element in each output vector.
target_ind = vec2ind(T_test);
test_ind = vec2ind(test_activity_type);

% Performance / training Visualisation : plots (commented out for convinience)
figure('Name','Error Rate','Color','white'), plotperform(tr)
figure('Name','Train State','Color','white'), plottrainstate(tr)
figure('Name','Confusion Mat','Color','white'), plotconfusion(T_test,test_activity_type)
%figure('Name','ROC curve : TEST set'), plotroc(T_test,test_activity_type)

% Calculate the classification accuracy rate (TEST SET)
classificationAccuracy = sum(target_ind == test_ind)/numel(target_ind);
Num_accurate = sum(target_ind == test_ind);
Num_misclassification = sum(target_ind ~= test_ind);
Num_totalnum = numel(target_ind);

fprintf('Test result : Hit %i out of %i sample | Test accuracy = %.5f\n',Num_accurate,Num_totalnum,classificationAccuracy)

