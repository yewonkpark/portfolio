%% 1.load the data
financial_data = readtable('ie_data_converted.xlsx');


%% 1.1. preprocess the data

%(a) refined the data to the periods without missing values
financial_data_refine = financial_data(121:1749,:);

%(b) take the complete annual cycle
%since there are only 9 months in 2016, we will exclude 2016
period = financial_data_refine(1:1620,:); %exclude 2016

%(c) check the period
num_months = size(period,1);
num_years = (size(period,1))/12; 

%(d) convert date to numeric values
date = table2array(period(:,1));
days = datenum(date);

%(e) convert 'days' cell to date type (for plotting)
days_type = datetime(days,'ConvertFrom','datenum');

%(f) data attributes
%we'll perform analysis on columns 8 to 10 : real price, real dividend, real earning
% 8th, 9th, 10th columns for real price, real dividend, real earning respectively

data = table2array(period(:,8:10)); 
data_name = financial_data_refine(:,8:10).Properties.VariableNames; %data attribute names

%% 1.2. Divide the data into train set and test set

%train set - the first 85 years (1881-1965) (1020 months)
train = data(1:1020,:);
train_month = days(1:1020,:);   %numeric type for date
train_date = days_type(1:1020); %datetime type for date (for plotting)

%test set - the remaining 50 years (1966-2015) (600 months)
test = data(1021:end,:);
test_month = days(1021:end,:);   %numeric type for date
test_date = days_type(1021:end); %datetime type for date (for plotting)

%% 2. Look for long term trends on a linear and log scale and de-trend the data first
% We will do that on the train set 

% 2.1. Apply Exponentially weighted moving average to see long term trend
alpha = 0.05;
exponentialMA = filter(alpha, [1 alpha-1], train);

%plot the train data and filtered data - upward trend
for k=1:3
    figure();
    plot (train_date, train(:,k), train_date, exponentialMA(:,k));
    axis tight
    legend('Train data', ...
       'Exponential Weighted Average','location','best')
    ylabel('Values')
    xlabel('Time')
    title(data_name{k})
end

%% 2.2. linear scale trend estimation 

% array to store the trend estimation (fitted line)
trend_line = zeros(size(train));

% trend estimation
for col = 1:3
    column= train(:,col);
    linear_coeffs = polyfit(train_month,column,1);
    trend_line(:,col) = polyval(linear_coeffs,train_month); 
end

% plot original(real) data and fitted estimation line

for k = 1:3
    figure();
    plot (train_date, train(:,k), train_date, trend_line(:,k));
    title (data_name{k})
    legend ('Original Data','Fitted Linear');
end

%% 2.3. log scale and fitted linear line

% array to store the trend estimation (fitted line)
trend_log = zeros(size(train));

% trend estimation
for col = 1:3
    column_log= log(train(:,col)); %take log of the original data
    log_coeffs = polyfit(train_month,column_log,1);
    trend_log(:,col) = polyval(log_coeffs,train_month); 
end

% plot original(real) data and fitted estimation line

for k = 1:3
    figure();
    plot (train_date, log(train(:,k)), train_date, trend_log(:,k));
    title (data_name{k})
    legend ('Log-scaled data','Fitted Line');
end

%% 2.4. Detrend data
% on linear scale

sdt_linear = zeros(size(train));
for col = 1:3
    column= train(:,col);
    linear_coeffs = polyfit(train_month,column,1); %make linear approximation
    linear_slope(:,col) = linear_coeffs(1);          %to get linear slope
    linear_intercept(:,col) = linear_coeffs(2);      %to get linear intercept
    %subtract linear approximation from data
    sdt_linear(:,col) = column - linear_intercept(:,col) - (train_month * linear_slope(:,col)); 
end

%plot original data vs detrend data
for k = 1:3
    figure();
    plot (train_date, train(:,k) ,train_date, sdt_linear(:,k));
    title (data_name{k})
    legend ('Original data','Detrend data');
end

% on log scale
sdt_log = zeros(size(train));
for col = 1:3
    column_log= log(train(:,col)); %take log of the original data
    linear_coeffs_log = polyfit(train_month,column_log,1); %make linear approximation
    linear_slope_log(:,col) = linear_coeffs_log(1);          %to get linear slope
    linear_intercept_log(:,col) = linear_coeffs_log(2);      %to get linear intercept
    %subtract linear approximation from log data
    sdt_log(:,col) = column_log - linear_intercept_log(:,col) - (train_month * linear_slope_log(:,col)); 
end

%plot log data vs detrend data
for k = 1:3
    figure();
    plot (train_date, log(train(:,k)), train_date, sdt_log(:,k));
    title (data_name{k})
    legend ('Log-scaled data','Detrend data');
end

%% 3. Calculate the FFT of the signal of interest, to estimate possible periodicities
% You can use a window to avoid spectral leaking

%let the detrended train set on the linear scale be data excerpt
excerptlen = size(train_month,1); %1020 months
excerpt = sdt_linear;

% create a hanning window
l = excerptlen
h = hanning(l); 
figure(), plot(h) % plot the window
spc = abs(fft(h)); % get the magnitude spectrum 
figure(), plot(spc) % ... and plot it

%% 3.1. fft - using CAPS for frequency domain 

EXCERPT = zeros(size(sdt_linear,1),size(sdt_linear,2))
for i = 1:size(sdt_linear,2)
    column = excerpt(:,i);
    FT = fft(h .* column);
    EXCERPT(:,i) = FT;   
end

%plot EXCERPT
abs_EXCERPT = zeros(size(sdt_linear,1),size(sdt_linear,2))
for i = 1:size(sdt_linear,2)
    figure()
    column = EXCERPT(:,i);
    abs_EXCERPT(:,i) = abs(column);
    plot(abs_EXCERPT(:,i));
    title (data_name{i})
end

%% 3.2. remove small components

%(a) take each column of the excerpt
a= EXCERPT(:,1);
b = EXCERPT(:,2);
c = EXCERPT(:,3);

%(b) set all coefficients smaller than a threshold to 0 for each column
%column 1 - Real Price
smallestfft_a =50; %set threshold
smallpos_a = find(abs(a) < smallestfft_a); %find small components
a(smallpos_a) = 0; %set small components = 0

%column 2 - Real Dividend
smallestfft_b =0.5;
smallpos_b = find(abs(b) < smallestfft_b);
b(smallpos_b) = 0;

%column 3 - Real Earning
smallestfft_c =1.0;
smallpos_c = find(abs(c) < smallestfft_c);
c(smallpos_c) = 0;

%(c) get the excerpt after getting rid of small components
PREDICT = zeros(size(EXCERPT));
PREDICT(:,1) = a;
PREDICT(:,2) = b;
PREDICT(:,3) = c;

%% 3.3. get predictions with ifft
predict = zeros(size(sdt_linear,1),size(sdt_linear,2));
for i = 1:size(sdt_linear,2)
    column = PREDICT(:,i);
    predict(:,i) = real(ifft(column));   
end

%% 3.4. add the trend back to the inversed signal 

%add the trend back
new_trend = zeros(size(predict));
for col = 1:3
    column_new= predict(:,col);
    new_trend(:,col) = column_new + linear_intercept(:,col) + (train_month * linear_slope(:,col)); 
end

%plot to compare with original data 
for k = 1:3
    figure();
    plot (train_date, train(:,k), train_date, new_trend(:,k));
    title (data_name{k})
    legend ('Original train data','Prediction');
end



%% 3.5. compare the test set with prediction
%take the first 600 months of the train prediction
%prediction of test set
predict_new = zeros(size(test,1),size(test,2));
for col = 1:size(new_trend,2)
    column=new_trend(:,col);
    predict_input = column(1:600);
    predict_new(:,col)=predict_input;
end

%% 3.6. get the trend for test data to add
% array to store the trend estimation (fitted line)
trend_line_test = zeros(size(test));

% trend estimation
for col = 1:3
    column= test(:,col);
    linear_coeffs = polyfit(test_month,column,1);
    trend_line_test(:,col) = polyval(linear_coeffs,test_month); 
end

% plot original(real) data and fitted estimation line

for k = 1:3
    figure();
    plot (test_date, test(:,k), test_date, trend_line_test(:,k));
    title (data_name{k})
    legend ('Original Test Data','Fitted Linear');
end

%% 3.7. add trend to the prediction
predict_new_final =  predict_new + trend_line_test

%% 3.8. prediction of the whole data
train_predict = [new_trend ; predict_new_final];

%plot original vs. predicted data
for k= 1:size(data,2)
    figure();
    hold on;
    plot(data(:,k),'g')
    plot(train(:,k), 'c');
    plot (train_predict(:,k), 'r');
    title (data_name{k});
    legend('Test','Train','Predict')
    hold off;
end

%% 4. Evaluating the test prediction using MSE

%calculate the difference between predicted values and original values
residuals = predict_new_final - test;

%MSE (for the whole data)
MSE_monthly = zeros(1,3);
for i=1:3
    residuals_column = residuals(:,i);
    MSE_monthly(:,i) = mean(abs(residuals_column).^2);
end

%make MSE piecewise
%split the residuals into 10-year intervals (5 equal parts)

splitarray = zeros(length(residuals),5);
residual_split = []
for i=0:4
   splitarray(i*120+1:(i+1)*120,i+1)=1;
   residuals(logical(splitarray(:,1)),:)
end

MSE = {}
for i=1:5
    residual_split{i} = residuals(logical(splitarray(:,i)),:)
    MSE{i} = mean(abs(residual_split{i}).^2); %calculate MSE for each part
end

%% 5. Calculating autocorrelation
%Reference: https://uk.mathworks.com/help/signal/ug/residual-analysis-with-autocorrelation.html

%Obtain the autocorrelation sequence of the residuals to lag 40

xc = zeros(81,3)
for i=1:3
    residuals_column = residuals(:,i)  
    xc(:,i) = xcorr(residuals_column,40,'coeff');
end

lags = linspace(-40,40,81); %create the lag

%Find the critical value for the 99%-confidence interval
conf99 = sqrt(2)*erfcinv(2*.01/2);
%Use the critical value to construct the lower and upper confidence bounds
lconf = -conf99/sqrt(length(test_month));
upconf = conf99/sqrt(length(test_month));

%Plot the autocorrelation sequence along with the 99%-confidence intervals
for k=1:3
    figure()
    stem(lags,xc(:,k),'filled')
    ylim([lconf-0.03 1.05])
    hold on
    plot(lags,lconf*ones(size(lags)),'r','linewidth',2)
    plot(lags,upconf*ones(size(lags)),'r','linewidth',2)
    title (data_name{k})
end

%The autocorrelation values clearly exceed the 99%-confidence bounds for a white noise autocorrelation at many lags.
%the model has not accounted for all the signal and therefore the residuals consist of signal plus noise.

%% 6. Refine your analysis

%Downsample the train set to 75 years (from 1891 - 1965)
refine1 = load('year1891-1965.mat'); %load the result

%Downsample the train set to 65 years (from 1901 - 1965)
refine2 = load('year1901-1965.mat'); %load the result

%Downsample the train set to 55 years (from 1911 - 1965)
refine3 = load('year1911-1965.mat'); %load the result


