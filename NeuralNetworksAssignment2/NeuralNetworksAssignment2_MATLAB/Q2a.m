clc;
clear;
close all;

% Generate data for training and testing
x_train = -1.6 : 0.05 : 1.6; 
y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train);
x_test = -1.6 : 0.01 : 1.6; 
y_test = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);

% The number of hidden neurons
hidden_neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100];

% Epochs set to 500 for training
epochs = 500;

for i = 1:length(hidden_neurons)

    % Convert from numeric arrays to cell arrays
    % Used by the adapt function for sequential mode training.
    x_cell = num2cell(x_train, 1);
    y_cell = num2cell(y_train, 1);

    n = hidden_neurons(i); % One hidden layer with n neurons
    net = feedforwardnet(n); % Define an MLP with n hidden neurons and SISO
    net.trainFcn = 'traingd'; % Regular gradient descent BP as training function
    net.performFcn = 'mse'; % Mean Square Error as performance function
    net.trainParam.lr = 0.01; % Learning rate set to 0.01
    net.trainParam.epochs = epochs; % Epochs set to 500

    % Sequential mode training
    for j = 1:epochs
        idx = randperm(length(x_train)); % Shuffle the order of training data to improve generalization ability
        net = adapt(net, x_cell(:,idx), y_cell(:,idx)); % Sample by sample training
    end

    % Using the trained MLP to predict the test set
    y_pred = net(x_test);

    % Plot the original function and the function inferred by the MLP
    figure;
    plot(x_test, y_test, 'r', 'LineWidth', 2);
    hold on;
    plot(x_test, y_pred, 'b--', 'LineWidth', 2);
    legend('True Function', 'MLP Approximation');
    xlabel('x');
    ylabel('y');
    title(['MLP Approximation with ', num2str(n), ' Hidden Neurons']);
    grid on;
    
    % Preidictions out of the training set
    x_extra = [-3, 3];
    y_extra_pred = net(x_extra);
    
    disp(['Hidden Neurons: ', num2str(n)]);
    disp('Ground truth for x=-3: 0.8090')
    disp(['MLP Prediction for x=-3: ', num2str(y_extra_pred(1))]);
    disp('Ground truth for x=3: 0.8090')
    disp(['MLP Prediction for x=3: ', num2str(y_extra_pred(2))]);
    disp('---------------------------------');
end