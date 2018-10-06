%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Code 03 : Parameter Experiment (Grid Search) for MLP           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note : PLEASE run the model sequentially! Do not clear the Workspace!
% This code is for demonstrating how we executed grid search

% This code might take long time to run : to shorten the time while also ensuring
% that the code is working, you can change the Val_repeat and
% net.trainParam.epochs with smaller number.


%% 1) sets of parameter for grid-search
% structure parameters
Num_neurons = {[20 20],[40 40],[80 80],[90 90],[100 100],[110 110],[150 150],[200 200],[20 6],...
[40 13] ,[80 27],[90 30],[100 33],[110 36],[150 50],[200 66]}; % two layers with [a, b] neurons for each layer respectively
activation_func = {{'tansig';'tansig';'softmax'},{'logsig';'logsig';'softmax'},{'logsig';'tansig';'softmax'},{'tansig';'logsig';'softmax'}}; %activation func

% learning parameters
learning_Rate =	[0.001, 0.01, 0.03, 0.05, 0.07 0.1]  ; %initial(base) learning rate
momentum = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; %momentum rate 

    
%% 2) experiment 1 : structure parameter (hiddenLayerSize & activation Func)

% we'll first do grid search for network structure parameters,
% (hiddenLayersize and activation function)
% find the best combination of parameters which generate the highest accuracy on validation set

% Creating a cell for storing the performance of each model

NN = size(Num_neurons,2); %rows for the cell (number of Num_neuron parameters)
AF = size(activation_func,2); %columns for the cell (number of learning_Rate parameters)
Experiment_accuracy_structure = zeros(NN,AF) %accuracy
Val_repeat = 3; %the number of repeat of grid-search (and average the val. accuracy)

for j = 1:Val_repeat;
    for i = 1:NN;
        for k = 1:AF;
            % train the MLP model with different parameter and get the accuracy grid
            fprintf('%i st iteration : now adding up accuracy rate.. average calculated at the end of the iteration',j);
            hiddenLayerSize = Num_neurons{i};
            
            net = patternnet(hiddenLayerSize, trainFcn);
            net.trainParam.epochs =	1000;
            net.trainParam.goal	= 0;
            net.trainParam.lr =	0.01;
            net.trainParam.lr_inc =	1.05;
            net.trainParam.lr_dec =	0.7;
            net.trainParam.max_perf_inc	= 1.04;
            net.trainParam.max_fail	= 10;
            net.trainParam.mc = 0.90;
            net.trainParam.min_grad =1e-5;
            net.layers.transferFcn = activation_func{k};
            net.performFcn = 'mse';
            net.trainParam.showCommandLine = false;
            net.trainParam.showWindow =	false;
            
            % Train the network and check the classfication accuracy on
            % validation set
            [net,tr] = train(net,x,t);
            
            y = net(x);
            
            tind = vec2ind(t);
            yind = vec2ind(y);
            
            tind_val = tind(valInd);
            yind_val = yind(valInd);
            
            ValidationAccuracy = sum(tind_val == yind_val)/numel(tind_val);
            
            % store the accuracy rate for each model
            Experiment_accuracy_structure(i,k) = (Experiment_accuracy_structure(i,k) + ValidationAccuracy)
        end       
    end
end

Experiment_accuracy_structure = Experiment_accuracy_structure/Val_repeat

%% 3) experiment 2 : learning rate & momentum

% now we apply the best combination of hiddenLayersize and activation 
% and execute the best learning rate & momentum grid-search

% Creating a cell for storing the performance of each model

LR = size(learning_Rate,2) %rows for the cell (number of Num_neuron parameters)
MM = size(momentum,2) %columns for the cell (number of learning_Rate parameters)
Experiment_accuracy_learning = zeros(LR,MM) %accuracy
Val_repeat = 3

for j = 1:Val_repeat;
    for i = 1:LR;
        for k = 1:MM;
            % train the MLP model with different parameter and get the accuracy grid
            % we feed the best structure parameters : [100 100], {'tansig';'tansig';'softmax'}
            % other parameters are all the same
            
            fprintf('%i st iteration : now adding up accuracy rate.. average calculated at the end of the iteration',j);
            hiddenLayerSize = [40 13];
            
            net = patternnet(hiddenLayerSize, trainFcn);
            net.trainParam.epochs =	1000;
            net.trainParam.goal	= 0;
            net.trainParam.lr =	learning_Rate(i);
            net.trainParam.lr_inc =	1.05;
            net.trainParam.lr_dec =	0.7;
            net.trainParam.max_perf_inc	= 1.04;
            net.trainParam.max_fail	= 10;
            net.trainParam.mc = momentum(k);
            net.trainParam.min_grad =1e-5;
            net.layers.transferFcn = {'tansig';'logsig';'softmax'};
            net.performFcn = 'mse';
            net.trainParam.showCommandLine = false;
            net.trainParam.showWindow =	false;

            % Train the network and check the classfication accuracy on
            % validation set
            [net,tr] = train(net,x,t);
            y = net(x);
            
            tind = vec2ind(t);
            yind = vec2ind(y);
            
            tind_val = tind(valInd);
            yind_val = yind(valInd);
            
            ValidationAccuracy = sum(tind_val == yind_val)/numel(tind_val);
            
            % store the accuracy rate for each model
            Experiment_accuracy_learning(i,k) = (Experiment_accuracy_learning(i,k) + ValidationAccuracy)
        end       
    end
end

Experiment_accuracy_learning = Experiment_accuracy_learning/Val_repeat



