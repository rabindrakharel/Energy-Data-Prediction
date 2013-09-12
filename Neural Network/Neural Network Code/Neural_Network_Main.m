%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Loading Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
data = csvread('dataset.csv');
data_size=length(data(:,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training Data Selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = data(1:round(0.6*data_size),1:5); 
y = data(1:round(0.6*data_size),6);
% i will need max and min of y to transform back predicted values

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Transformation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_y = max(y);
min_y = min(y);

% let me normalize y in range 0.25 to 0.75 in this range sigmoid is linear
y = (y - min_y)./(max_y - min_y) * 0.6 + 0.2;
load train
% X is normalized input data set with 5 columns
% I will form the 2 layer networks with 10 neurons in each
[m, n] = size(X);
for i=1:5
    X(:,i) = (X(:,i) - min(X(:,i)))./(max(X(:,i)) - min(X(:,i)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Neural Network Formation and Training %%%%%%%%%%%%%%%

% initialie weights to small random number
hidden_layer_size = 10;
input_layer_size = n;
% initialization to small random number between [-epsilon, epsilon]
% where epsilon = sqrt(6)/#hidden_layer_size
epsilon = 2 * 6.^0.5./hidden_layer_size;
Theta1 = (rand(hidden_layer_size, (input_layer_size + 1)) -0.5)*epsilon;
Theta2 = (rand(1, (hidden_layer_size + 1)) - 0.5)*epsilon;
% size(Theta1)
% size(Theta2)
% and unrolling them to a vector so that our optim function can use it
nn_params_initial = [Theta1(:) ; Theta2(:)];
[J, D] = cost(nn_params_initial,hidden_layer_size, X, y, 0.1);
fprintf(' Cost for initialized weights = %.3f\n', J);
% setup for optimization
options = optimset('MaxIter',150);
lambda = 0.001; % The regularization term
% handle to the matlab function
costFunction = @(p) cost(p, hidden_layer_size,X, y, lambda);
 
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, nn_params_initial, options);
save nn_params
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));
				 
				 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Cross Validating Neural network in 20% data %%%%%
x1lim=round(0.6*data_size)
x2lim=round(0.8*data_size)
CX = data(x1lim:x2lim,1:5); 
for i=1:5
    CX(:,i) = (CX(:,i) - min(CX(:,i)))./(max(CX(:,i)) - min(CX(:,i)));
end
Cy = data(x1lim:x2lim,6);
p = predict(Theta1,Theta2,CX, max_y, min_y);
% checking the correctness
plot(Cy(1:50,:), 'r');hold on;plot(p(1:50,:), 'g');
legend({'original', 'predicted'});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Testing in 20% more data %%%%%%%%%%%%%%%%%%%%%%%%
x1lim=round(0.8*data_size)
x2lim=round(1*data_size)
TX = data(x1lim:x2lim,1:5); 
for i=1:5
    TX(:,i) = (TX(:,i) - min(TX(:,i)))./(max(TX(:,i)) - min(TX(:,i)));
end
Ty = data(x1lim:x2lim,6);
p = predict(Theta1,Theta2,TX, max_y, min_y);
MAE=mean(abs(p-Ty))
