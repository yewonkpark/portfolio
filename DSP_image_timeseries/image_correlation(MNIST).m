%% 1-1) Read the training and testset
% the dataset is MNIST dataset
[images_train, digits_train] = readImgTxt('training.txt');
[images_test, digits_test] = readImgTxt('test.txt');

%% 1-2) Preprocessing 

% number of data samples (images) : will be used repeatedly
size_test = size(images_test,1);
size_train = size(images_train,1);

% here create cell to store images without offset
images_test_woff = cell(size_test,1); 
images_train_woff = cell(size_train,1);

for i = 1:size_test
    images_test_woff{i} = images_test{i} - mean2(images_test{i});
end

for j = 1:size_train
    images_train_woff{j} = images_train{j} - mean2(images_train{j});
end
    
%% 2. calculate correlation
% we created a function for comparing each test image with the collection of training images
% using the correlation : Corr_Calculation()

% here create array to store calculated correlation between images without offset
store_corr = zeros(size_test, size_train);
store_corr = Corr_Calculation(images_test_woff,images_train_woff,store_corr);

%% 3. Estimating the label for test image by matching training images with max correlation
% 3-1) For each test image, match the training images with the highest correlation coefficient

Index_HighestCorr = zeros(size_test,1); %create array to store the result

for k = 1:size_test % for every test image
    % pick the value and location (index) of training image which has highest corr. coefficient
    [Max, Index] = max(store_corr(k,:)); 
    % convert the location (linear index) of traning images in the right format
    [I_row, I_col] = ind2sub([1,size_train],Index);
    % I_col is the index number of training images best match the test image
    % that is, k = 3, I_col = 1633 indicates that 3rd test image has
    % highest correlation coefficient with 1633th training image
    Index_HighestCorr(k) = I_col; %store the result
end

%% 3-2) Now we will create the matrix contains labels(digits) for these test and train images

digits_compare = zeros(size_test,2); %create array to compare labels
digits_compare(:,1) = digits_test;

% by referring index we get in 3-1), find and assign the label of train image 
for o = 1: size_test
    digits_compare(o,2) = digits_train(Index_HighestCorr(o));
end


%% 4. Calculate the accuracy of digit recognition system
accurate_matching = sum(digits_compare(:,1) == digits_compare(:,2));
accuracy_rate = accurate_matching/size_test;
fprintf('The accuracy rate of system = %.2f\n',accuracy_rate);

%% 5. Calculate 2D-corr2 (xcorr2) to find the best match images
% used original image here 
% (need to make sure we have to use images without offset)

% calculate 2D-corr2 across all images combinations
store_xcorr = cell(size_test, size_train);
for i = 1:size_test
    for j = 1:size_train
        store_xcorr{i,j} = xcorr2(images_test{i},images_train{j});
    end
end

% calculate the mean of xcorr2
mean_xcorr = zeros(size_test, size_train);
for i = 1:size_test
    for j = 1:size_train
        mean_xcorr(i,j) = mean(mean(store_xcorr{i,j}));
    end
end

% Now, find the images with maximum correlation to find the best match
[Max, Index] = max(mean_xcorr(:));
[I_row, I_col] = ind2sub(size(mean_xcorr),Index);
fprintf('The best match images is : %i th test image and %i th training image',I_row,I_col);

% plot the best-match two images
figure(), imshow(images_test{I_row})
figure(), imshow(images_train{I_col})

%% 6. Implement variants of digit recognition system
% we converted/transformed the images based on images without offset

% 6-1) use negative of both test and train images

% create cell to store negatives of images 
images_test_ng = cell(size_test,1); 
images_train_ng = cell(size_train,1);

% transform images by using incomplement(I)
for i = 1:size_test
    images_test_ng{i} = imcomplement(images_test_woff{i});
end

for j = 1:size_train
    images_train_ng{j} = imcomplement(images_train_woff{j});
end
    

% repeat the stepes 2, 3-1), 3-2), 4 
% comments are omitted here aviod repetation/redundancy ; previous explained in each step

store_corr_ng = zeros(size_test, size_train);
store_corr_ng = Corr_Calculation(images_test_ng,images_train_ng,store_corr_ng);

Index_HighestCorr_ng = zeros(size_test,1); 

for k = 1:size_test
    [Max, Index] = max(store_corr_ng(k,:)); 
    [I_row, I_col] = ind2sub([1,size_train],Index);
    Index_HighestCorr_ng(k) = I_col; 
end

digits_compare_ng = zeros(size_test,2); 
digits_compare_ng(:,1) = digits_test;

for o = 1: size_test
    digits_compare_ng(o,2) = digits_train(Index_HighestCorr_ng(o));
end

%% 6-2) use slightly rotated images for training set

% create cell to store rotated training images  
images_train_rotate = cell(size_train,1);

% transform images by using imrotate(I, angle)
angle = 10;  % can change the angle 
for j = 1:size_train
    images_train_rotate{j} = imrotate(images_train_woff{j}, angle, 'crop');
    % here 'crop' bbox used to maintain the same size with test image
end

% repeat the stepes 2, 3-1), 3-2), 4 
% comments are omitted here aviod repetation/redundancy ; previous explained in each step

store_corr_rt = zeros(size_test, size_train);
store_corr_rt = Corr_Calculation(images_test_woff,images_train_rotate,store_corr_rt);

% Index_HighestCorr_rt = zeros(size_test,1); 

for k = 1:size_test
    [Max, Index] = max(store_corr_rt(k,:)); 
    [I_row, I_col] = ind2sub([1,size_train],Index);
    Index_HighestCorr_rt(k) = I_col; 
end

digits_compare_rt = zeros(size_test,2); 
digits_compare_rt(:,1) = digits_test;

for o = 1: size_test
    digits_compare_rt(o,2) = digits_train(Index_HighestCorr_rt(o));
end

%% 6-3) use training images with some noises

% create cell to store training images with noise  
images_train_noise = cell(size_train,1);

% transform train images by using imnoise(I, 'salt & pepper', noise)
noise = 0.2;  % can change the noise level
for j = 1:size_train
    images_train_noise{j} = imnoise(images_train_woff{j}, 'salt & pepper', noise);
    % here 'salt & pepper' noise used
end


% repeat the stepes 2, 3-1), 3-2), 4 
% comments are omitted here aviod repetation/redundancy ; previous explained in each step

store_corr_ns = zeros(size_test, size_train);
store_corr_ns = Corr_Calculation(images_test_woff,images_train_noise,store_corr_ns);

Index_HighestCorr_ns = zeros(size_test,1); 

for k = 1:size_test
    [Max, Index] = max(store_corr_ns(k,:)); 
    [I_row, I_col] = ind2sub([1,size_train],Index);
    Index_HighestCorr_ns(k) = I_col; 
end

digits_compare_ns = zeros(size_test,2); 
digits_compare_ns(:,1) = digits_test;

for o = 1: size_test
    digits_compare_ns(o,2) = digits_train(Index_HighestCorr_ns(o));
end

%% 6-4) this part is just for visualise comparison of transformed images
close all;
imageno = 490
figure(), imshow(images_train{imageno}) %original
figure(), imshow(images_train_woff{imageno}) %without offset
figure(), imshow(images_train_ng{imageno}) %negatives
figure(), imshow(images_train_rotate{imageno}) %rotated
figure(), imshow(images_train_noise{imageno}) %add noise

%% 7. Compare the accuracy rate of different system 

accurate_matching_ng = sum(digits_compare_ng(:,1) == digits_compare_ng(:,2));
accuracy_rate_ng = accurate_matching_ng/size_test;
fprintf('The accuracy rate of system (negatives) = %.3f\n',accuracy_rate_ng);

accurate_matching_rt = sum(digits_compare_rt(:,1) == digits_compare_rt(:,2));
accuracy_rate_rt = accurate_matching_rt/size_test;
fprintf('The accuracy rate of system (rotated) = %.3f\n',accuracy_rate_rt);

accurate_matching_ns = sum(digits_compare_ns(:,1) == digits_compare_ns(:,2));
accuracy_rate_ns = accurate_matching_ns/size_test;
fprintf('The accuracy rate of system (noise added)= %.3f\n',accuracy_rate_ns);