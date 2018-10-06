
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Code 05 : Parameter Experiment (Grid Search) for SAE            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note : PLEASE run the model sequentially! Do not clear the Workspace!
% This code is for demonstrating how we executed grid search

% [WARNING] This code takes LONG time to run : to shorten the time while also ensuring
% that the code is working, you can change the maximum number of epochs to smaller number
% ('MaxEpochs',deep_autoencoder.trainParam.epochs for all 4 layers)


%% 1) sets of parameter for grid-search
% structure parameters
Num_neurons_1 = {10, 50, 100, 300,600,1000,2000}; % for the first AE
Num_neurons_2 = {10, 50, 100, 300,600,1000,2000}; % for the second AE

% learning parameters
Sparsity_prop = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3]; %sparsity target
Sparsity_reg = [0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]; %sparsity regulariser coefficient


%% 2) experiment 1 : structure parameter

% we'll first do grid search for network structure parameters 
% (the number of units for each layer)
% find the best combination of parameters which generate the highest accuracy on validation set

% Creating a cell for storing the performance of each model
NN1 = size(Num_neurons_1,2); %rows for the cell (number of Num_neuron parameters)
NN2 = size(Num_neurons_2,2); %columns for the cell (number of learning_Rate parameters)
Experiment_accuracy_structure2 = zeros(NN1,NN2) %accuracy
Val_repeat = 1; %the number of repeat of grid-search (and average the val. accuracy)

for j = 1:Val_repeat;
    for i = 1:NN1;
        for k = 1:NN2;
            % train the SAE model with different parameter and get the accuracy grid
            fprintf('%i st iteration : now adding up accuracy rate.. average calculated at the end of the iteration',j);
            deep_autoencoder=init(deep_autoencoder);%initialise the autoencoder for retraining
            
            % PRE-TRAINING phase
            % train the first autoencoder
            hiddenSize_ae1 = Num_neurons_1{i};
            autoenc_ae1 = trainAutoencoder(X_train,hiddenSize_ae1,...
                'MaxEpochs',1000,... 
                'EncoderTransferFunction','logsig',... 
                'DecoderTransferFunction','logsig',... 
                'ScaleData',true,...
                'L2WeightRegularization',0.001,... 
                'SparsityRegularization',1,... 
                'SparsityProportion',0.05);
    
            % Extract the features in the hidden layer -> to feed into the second auto-encoder!
            features_ae1 = encode(autoenc_ae1,X_train);
            
            % train the second autoencoder using the features from the first autoencoder
            
            hiddenSize_ae2 = Num_neurons_2{k};
            autoenc_ae2 = trainAutoencoder(features_ae1,hiddenSize_ae2,...
                'MaxEpochs',1000,... 
                'EncoderTransferFunction','logsig',... 
                'DecoderTransferFunction','logsig',... 
                'ScaleData',true,...
                'L2WeightRegularization',0.001,... 
                'SparsityRegularization',1,... 
                'SparsityProportion',0.05);
                     
            % Extract the features in the hidden layer -> to feed into the output layer
            features_ae2 = encode(autoenc_ae2,features_ae1);

            % FINE-TUNING phase
            
            softnet = trainSoftmaxLayer(features_ae2,T_train,'LossFunction','mse','MaxEpochs',1000);
            deep_autoencoder = stack(autoenc_ae1,autoenc_ae2,softnet);
            % feed the training data and train the classifier
            deep_autoencoder.trainParam.epochs = 1000;	
            deep_autoencoder = train(deep_autoencoder,X_train,T_train);
            
            % Validation Accuracy
            validation_activity_type = deep_autoencoder(X_val);
            val_target = vec2ind(T_val);
            val_result = vec2ind(validation_activity_type);
            ValidationAccuracy = sum(val_target == val_result)/numel(val_target);
            % store the accuracy rate for each model
            Experiment_accuracy_structure2(i,k) = (Experiment_accuracy_structure2(i,k) + ValidationAccuracy)
        end
    end
end

Experiment_accuracy_structure2 = Experiment_accuracy_structure2/Val_repeat

%% 3) experiment 2 : learning parameter (sparsity)

% now we apply the best combination of the number of hidden neurons
% and execute the sparsity learinng parameter grid-search


% find the best combination of parameters which generate the highest accuracy on validation set

% Creating a cell for storing the performance of each model

SP = size(Sparsity_prop,2); %rows for the cell (number of Num_neuron parameters)
SR = size(Sparsity_reg,2); %columns for the cell (number of learning_Rate parameters)
Experiment_accuracy_sparsity = zeros(SP,SR) %accuracy
Val_repeat = 1; %the number of repeat of grid-search (and average the val. accuracy)

for j = 1:Val_repeat;
    for i = 1:SP;
        for k = 1:SR;
            % train the SAE model with different parameter and get the accuracy grid
            % we feed the best structure parameters : [10 10]
            % other parameters are all the same
            fprintf('%i st iteration : now adding up accuracy rate.. average calculated at the end of the iteration',j);
            deep_autoencoder=init(deep_autoencoder); %initialise the autoencoder for retraining
            % PRE-TRAINING phase
            % train the first autoencoder
            hiddenSize_ae1 = 10;
            autoenc_ae1 = trainAutoencoder(X_train,hiddenSize_ae1,...
                'MaxEpochs',1000,... 
                'EncoderTransferFunction','logsig',... 
                'DecoderTransferFunction','logsig',... 
                'ScaleData',true,...
                'L2WeightRegularization',0.001,... 
                'SparsityRegularization',Sparsity_reg(k),... 
                'SparsityProportion',Sparsity_prop(i));
    
            % Extract the features in the hidden layer -> to feed into the second auto-encoder!
            features_ae1 = encode(autoenc_ae1,X_train);
            
            % train the second autoencoder using the features from the first autoencoder
            
            hiddenSize_ae2 = 10;
            autoenc_ae2 = trainAutoencoder(features_ae1,hiddenSize_ae2,...
                'MaxEpochs',1000,... 
                'EncoderTransferFunction','logsig',... 
                'DecoderTransferFunction','logsig',... 
                'ScaleData',true,...
                'L2WeightRegularization',0.001,... 
                'SparsityRegularization',Sparsity_reg(k),... 
                'SparsityProportion',Sparsity_prop(i));
                     
            % Extract the features in the hidden layer -> to feed into the output layer
            features_ae2 = encode(autoenc_ae2,features_ae1);

            % FINE-TUNING phase
            
            softnet = trainSoftmaxLayer(features_ae2,T_train,'LossFunction','mse','MaxEpochs',1000);
            deep_autoencoder = stack(autoenc_ae1,autoenc_ae2,softnet);
            % feed the training data and train the classifier
            deep_autoencoder.trainParam.epochs = 1000;	
            deep_autoencoder = train(deep_autoencoder,X_train,T_train);
            
            % Validation Accuracy
            validation_activity_type = deep_autoencoder(X_val);
            val_target = vec2ind(T_val);
            val_result = vec2ind(validation_activity_type);
            ValidationAccuracy = sum(val_target == val_result)/numel(val_target);
            % store the accuracy rate for each model
            Experiment_accuracy_sparsity(i,k) = (Experiment_accuracy_sparsity(i,k) + ValidationAccuracy)
        end
    end
end

Experiment_accuracy_sparsity = Experiment_accuracy_sparsity/Val_repeat
