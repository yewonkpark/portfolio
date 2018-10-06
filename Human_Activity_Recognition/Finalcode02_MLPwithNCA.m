
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Code 02 : Multilayer Perceptron (MLP) Neural Network                  %
%                      with feature selection (NCA)                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note : PLEASE run the model sequentially! Do not clear the Workspace!
% Here we feed the model with the best set of parameters (optimised) 
% which was experimented with 'Finalcode03_MLPgridsearch'


%% 1. Create the Network and set parameters

% 1) Set the network structure parameters 

% choose training function
   
trainFcn = 'traingdx';  % Gradient descent w/momentum & adaptive lr backpropagation

% choose the Num of neuorns in the hidden layer
% two layers with [a, b] neurons for each layer respectively
hiddenLayerSize = [40 13]; 

% 2) Create a Pattern Recognition Network

% create network
net = patternnet(hiddenLayerSize, trainFcn); %patternnet for classification

% 3) Set the Learning Parameters and Stopping Condition

% --Maximum number of epochs to train. training stops when reached.
net.trainParam.epochs =	1000;	

% --Performance (error rate) goal. training stops when reached.
net.trainParam.goal	= 0;	

% --Learning rate (base)
net.trainParam.lr =	0.07;
% Ratio to increase Lr : For each epoch, if performance decreases toward the goal, 
% then the learning rate is increased by the factor lr_inc
net.trainParam.lr_inc =	1.05;

% Ratio to decrease Lr : If performance increases by more than the factor max_perf_inc, 
% Lr is decreasd by the factor lr_dec 
net.trainParam.lr_dec =	0.7;	

% Maximum performance increase
net.trainParam.max_perf_inc	= 1.04;

% Maximum validation failures : Training stops when validation performance
% (error rate) has increased more than max_fail times since the last time it decreased
% act as 'Early stopping' 

net.trainParam.max_fail = 10;

% Momentum
net.trainParam.mc = 0.4; %Momentum constant
net.trainParam.min_grad =1e-5; 	%Minimum performance gradient


% 4) transfer function = activiation function for each layer
% help transfer for lists of function : the last one should be softmax
% (classification)
net.layers.transferFcn = {'tansig';'logsig';'softmax'};


% 5) Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'divideind';  % Divide data by pre-randomly selected index (code01)
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;


% 6) Choose a Performance Function
% For a list of all performance functions type: help nnperformance
% chose MSE : the mean of squared errors.
net.performFcn = 'mse';  

% 7) Plot related setting
% For a list of all plot functions type: help nnplot
% picked the plot seems necessary to look at
net.plotFcns = {'plotperform','plottrainstate','plotroc','plotconfusion'};

net.trainParam.show = 5;% Epochs between displays (NaN for no displays)
net.trainParam.showCommandLine = true;	%Generate command-line output
net.trainParam.showWindow =	true;	%Show training GUI


%% 2. Train and Test the network

% initialise weights & bias and retrain the network
net = init(net);
% Train the Network
[net,tr] = train(net,x,t); 

% Test the Network
y = net(x);

% Perform vec2ind()
% The network outputs will be in the range 0 to 1, so vec2ind function used
% to get the class indices as the position of the highest element in each output vector.
tind = vec2ind(t);
yind = vec2ind(y);

tind_test = tind(testInd);
yind_test = yind(testInd);

% Calculate the classification accuracy rate (TEST SET)
classificationAccuracy = sum(tind_test == yind_test)/numel(tind_test);
Num_accurate = sum(tind_test == yind_test);
Num_misclassification = sum(tind_test ~= yind_test);
Num_totalnum = numel(tind_test);
fprintf('Test result : Hit %i out of %i sample | Test accuracy = %.5f\n',Num_accurate,Num_totalnum,classificationAccuracy)


% Performance / training Visualisation : plots (commented out for convinience)
figure('Name','Error Rate','Color','white'), plotperform(tr)
figure('Name','Train State','Color','white'), plottrainstate(tr)
figure('Name','Confusion Mat','Color','white'), plotconfusion(t(:,testInd),y(:,testInd))
%figure('Name','ROC curve : TEST set'), plotroc(t(:,testInd),y(:,testInd))

%% 4. Performance evaluation

% If model was successfully trained at 3, 
% Initialise/Retrain the network 5 times and store the performance metrics
% Generate the average metrics and viz.

Num_of_run = 5; %the number of time to train the model
Store_model = cell(Num_of_run,1); %cell for storing the trained model
Store_accuracy = zeros(Num_of_run,1); %cell for storing the accuracy rate
Store_label = zeros(Num_of_run,numel(tind_test)); %cell for storing the prediction

for i = 1:Num_of_run; 
    net.trainParam.showCommandLine = true;	% command line control  
    % initialise weights & bias and retrain the network
    net = init(net);
    Store_model{i} = train(net,x,t); %will store the trained model
    % test the network
    yi = Store_model{i}(x(:,testInd));
    % Perform vec2ind()  
    yi_ind = vec2ind(yi);
    Store_label(i,:) = yi_ind
    
    % calculate and store the classification accuracy rate on test set
    classificationAccuracy2 = sum(tind_test == yi_ind)/numel(yi_ind);
    Store_accuracy(i) = classificationAccuracy2
    Num_accurate2 = sum(tind_test == yi_ind);
    Num_totalnum2 = numel(yi_ind);
    fprintf('Test result _repeat %i : Hit %i out of %i sample | Test accuracy = %.5f\n',i,Num_accurate2,Num_totalnum2,classificationAccuracy2);

end


Average_Test_Accuracy_MLP = mean(Store_accuracy(:,1))


