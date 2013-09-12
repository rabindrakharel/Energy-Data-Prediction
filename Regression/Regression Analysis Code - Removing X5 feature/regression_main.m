%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear ; close all; clc
%% Load Data
%  The first 5 columns contains the X values and the 6th column
%  contains the label (y).
data = csvread('dataset.csv');
data_size=length(data(:,1));
%60 data is for training purpose
X = data(1:round(0.6*data_size),1:5); 
y = data(1:round(0.6*data_size),6);
% i will need max and min of y to transform back predicted values
max_y = max(y);
min_y = min(y);
% let me normalize y in range 0.25 to 0.75 in this range sigmoid is linear
y = (y - min_y)./(max_y - min_y) * 0.6 + 0.2;
XF= mapFeature(X(:,3),X(:,4));
X=[X(:,1) X(:,2)  XF(:,2:end)];
% Normalize X values
[X mu sigma] = featureNormalize(X);
X(1,:)
m=length(X(:,1));
X=[ones(m,1) X];
initial_theta = zeros(size(X,2),1);
% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0.001;
% Set Options
options = optimset('MaxIter', 500);
% Optimize
[theta, J, exit_flag] = ...
	fmincg(@(t)costFunctionReg(t, X, y, lambda), initial_theta, options);
% Compute accuracy on our training set
theta % This is the saturated value of theta which will be further used for prediction


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Cross Validation Phase   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
x1lim=round(0.6*data_size)
x2lim=round(0.8*data_size)
CX = data(x1lim:x2lim,1:5); 
%CX1 = CX(:,1:5);
CXF= mapFeature(CX(:,3), CX(:,4));
CX=[CX(:,1) CX(:,2) CXF(:,2:end)];
Cy = data(x1lim:x2lim,6);
[CX mu sigma] = featureNormalize(CX);
CX=[ones(size(CX,1),1) CX];
CY=sigmoid(CX*theta); % output values are obtained for cross validation set
Y = (CY - 0.2)./0.6 .* (max_y - min_y) + min_y; % Output are transfromed to real output values
plot(Cy(1:50,:), 'r');hold on;plot(Y(1:50,:), 'g')
legend({'original', 'predicted'});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Testing Phase   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
TX = data(round(0.8*data_size):round(1*data_size),1:5); 
Ty = data(round(0.8*data_size):round(1*data_size),6);
%TX1 = TX(:,1:5);
TXF= mapFeature(TX(:,3), TX(:,4));
TX=[TX(:,1) TX(:,2) TXF(:,2:end)];
[TX mu sigma] = featureNormalize(TX);
TX=[ones(size(TX,1),1) TX];
TY=sigmoid(TX*theta); % output values are obtained for cross validation set
Y = (TY - 0.2)./0.6 .* (max_y - min_y) + min_y; % Output are transfromed to real output values
MAE=mean(abs(Y-Ty))