%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Code 01 : Data Preparation for two Neural Network models         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dataset is downloadable at https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
% Note : PLEASE run the model sequentially! (Code 01 -> 02 -> ... -> 05)

%% 1. Clear up and Load/preprocess the data

% 1) Clear the workspace, command, figures
clc; clear all; close all; 

% 2) Load the data : we merge the originally provided train/test data 
% Later, repartition into train/validation/test set by hold-out method

rawdata_attribute = importdata('X_train.txt');
rawdata_label = importdata('y_train.txt');
rawdata2_attribute = importdata('X_test.txt');
rawdata2_label = importdata('y_test.txt');

% Whole data merged here
totaldata_attribute = vertcat(rawdata_attribute, rawdata2_attribute);
totaldata_label = vertcat(rawdata_label, rawdata2_label);

%% 2. Feature selection using neighbourhood component analysis (NCA)
% this part is for our first model (MLP + feature selection using NCA)
% the NCA method of feature selection for it's ease of implementation in
% Matlab, please refer to the report for full details. 

%data & labels are fit to the NCA algorithm, using stochastic gradient
%descent for optimisation 
mdl = fscnca(totaldata_attribute,totaldata_label,'Solver','sgd');

%feature weights & indexes visualised
figure();
plot(mdl.FeatureWeights,'ro');
grid on;
xlabel('Feature index');
ylabel('Feature weight');

% threshold value set - all features weights < 0.02 will be removed from the
% dataset, prior to loading into the input layer of MLP
threshold = 0.02; % a value of slightly > 0 was chosen to remove irrelvant features
above_threshold = find(mdl.FeatureWeights > threshold*max(1,max(mdl.FeatureWeights)));

%this is the new set of features(attributes) for MLP model
features = totaldata_attribute(:,above_threshold);

%% 3. Transform discrete label to generate the target vector 

% take the number of label(class) = 6
labNum = size(unique(totaldata_label),1); 

% apply the predefined function to convert discrete label to target vectors
% here we get matrix of 1 and 0 displaying class label for each row
% (e.g.) lable 3 : [0 0 1 0 0 0]
totaldata_label_final = discrete2targetvec(totaldata_label,labNum);

%% 4. generate the final partitioned dataset 

%training/ validation / test dataset ratio : 0.7, 0.15, 0.15
[trainInd,valInd,testInd] = dividerand(size(totaldata_label,1),0.7,0.15,0.15);

%1) for MLP model (after feature selection)
%x = features'; % input data (attributes)
%t = totaldata_label_final'; % output target data (labels)

%2) for SAE model

X_train = totaldata_attribute(trainInd,:)'; % input data (attributes) : trainset
T_train = totaldata_label_final(trainInd,:)'; % output target data (labels) :trainset
X_val = totaldata_attribute(valInd,:)'; % input data (attributes) : validation set
T_val = totaldata_label_final(valInd,:)'; % output target data (labels) :validation set
X_test = totaldata_attribute(testInd,:)'; % input data (attributes) : testset
T_test = totaldata_label_final(testInd,:)'; % output target data (labels) :testset

%% network test
csvwrite('networktest_X_test.csv',X_test)
csvwrite('networktest_X_train.csv',X_train)
csvwrite('networktest_Y_test.csv',T_test)
csvwrite('networktest_Y_train.csv',T_train)