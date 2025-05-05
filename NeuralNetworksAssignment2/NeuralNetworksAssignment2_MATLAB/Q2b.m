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

for i = 1:length(hidden_neurons)
    n = hidden_neurons(i); % One hidden layer with n neurons

    net = feedforwardnet(n); % Define an MLP with n hidden neurons and SISO
    net.trainFcn = 'trainlm'; % Levenberg-Marquardt BP as training function, no learning rate involved
    net.performFcn = 'mse'; % Mean Square Error as performance function
    
    % Train the MLP
    net = train(net, x_train, y_train); % Default epochs set to 1000 and use batch mode
    
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
