clc
clear all
%% LOADING THE DATA
% the dataset is downloadable at https://archive.ics.uci.edu/ml/datasets/skin+segmentation
rawdata = importdata('skin.csv'); %load the data
skindata = array2table(rawdata); %convert to table for cvpartition
skindata.Properties.VariableNames = {'Blue','Green','Red','Skinclass'}; %assign atrribute name

%% BASIC STATISTICS
%Initial observation of the distribution of each attribute & class

%P(class)
classraw = categorical(rawdata(:,4));
countclass = countcats(classraw);
p_skin = countclass(1)/size(rawdata,1)
p_nskin = countclass(2)/size(rawdata,1)
red = rawdata(:,1);
green = rawdata(:,2);
blue = rawdata(:,3);

%Plot the distribution of each attribute
figure('Name','Initial observation')
subplot(3,1,1)
histogram(red,250)
title('Blue')
subplot(3,1,2)
histogram(green,250)
title('Green')
subplot(3,1,3)
histogram(blue,250)
title('Red')

% 3D classifier distribution
% we can rotate this graph! (tool -> span). see how 'skin' class are
% heavily clustered at the center

graphskin = importdata('graphskin.csv'); 
graphnskin = importdata('graphnskin.csv');

figure('Name','Initial observation_3D classfier distribution')
scatter3(graphskin(:,1), graphskin(:,2), graphskin(:,3),30,'blue');
hold on;
scatter3(graphnskin(:,1), graphnskin(:,2), graphnskin(:,3),30,'red');
xlabel('Blue')
ylabel('Green')
zlabel('Red')
legend('skin','non-skin')

%% PARTITION OF THE DATA
% since we have enough number of data (245,057 in total), here we do hold-out validation

skincv = cvpartition(skindata.Skinclass, 'HoldOut', 0.30); % leave out 30% for test, 70% for training
idxTrain = training(skincv); 
idxTest = test(skincv); 
dataTrain = skindata(idxTrain,:); %training data
dataTest = skindata(idxTest,:); %test data


%% RANDOM FOREST Best Model
% Model with best performance proven by /derived from the code03_randomforestfitting.m

Tree_Best = templateTree('MaxNumSplits',20, 'MinLeafSize',200,'MinParentSize',2) %Tree condition
RF_Best = fitcensemble(dataTrain, 'Skinclass','Method','GentleBoost','NumLearningCycles',50,'Learners',Tree_Best); %Best model

% Prediction for test data and measurement of computation time
tic;
prediction_RFBest = predict(RF_Best, dataTest);
Time_RF = toc


% Evaluation : error-rate for Random Forest
testErrRF = loss(RF_Best, dataTest)

% Generating confusion matrix 
[resultRF,classRF] = confusionmat(dataTest.Skinclass, prediction_RFBest)

% Number of misclassification
missclassificationRF = resultRF(1,2) + resultRF(2,1)

% Plot the classification result : the confusion matrix
figure('Name','True/Predicted class matrix RF') 
bar3(resultRF)
legend('skin','non-skin');
xlabel('Predicted result');
ylabel('True class');

%% KNN Best Model
% Model with best performance proven by /derived from from the code02_knnfitting.m
KNN_Best = fitcknn(dataTrain, 'Skinclass','NSMethod','kdtree','Distance','euclidean','NumNeighbors',3);

% Prediction for test data and measurement of computation time
tic;
prediction_KNNBest = predict(KNN_Best, dataTest);
Time_KNN = toc

% Evaluation : error-rate for KNN
testErrKNN = loss(KNN_Best, dataTest)

% Generating confusion matrix 
[resultKNN,classKNN] = confusionmat(dataTest.Skinclass, prediction_KNNBest) 

% Number of misclassification
missclassificationKNN = resultKNN(1,2) + resultKNN(2,1)

% Plot the classification result : the confusion matrix
figure('Name','True/Predicted class matrix KNN')
bar3(resultKNN)
legend('skin','non-skin');
xlabel('Predicted result');
ylabel('True class');