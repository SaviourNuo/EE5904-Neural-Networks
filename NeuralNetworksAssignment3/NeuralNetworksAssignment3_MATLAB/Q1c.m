clc; clear; close all;

%% Generate training and test data
x_train = -1.6:0.08:1.6; % Training set with a uniform step length 0.08
y_train = 1.2*sin(pi*x_train) - cos(2.4*pi*x_train) + 0.3*randn(size(x_train)); % Add random noise
x_test = -1.6:0.01:1.6; % Test set with a uniform step length 0.01
y_test = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test); % Ground truth without noise

%% Construct RBF Kernel Matrix for Training Data
N = length(x_train); % Number of training points
sigma = 0.1; % Standard deviation for Gaussian RBF
Phi_train = zeros(N, N); % Initialize Gaussian Kernel

for i = 1:N
    for j = 1:N
        Phi_train(i, j) = exp(-((x_train(i) - x_train(j))^2) / (2 * sigma^2)); % Gaussian Kernel
    end
end

%% Calculate the weights under different regularization parameters λ
lambda_values = [0, 1e-6, 1e-4, 1e-2, 1, 10, 40]; % Different λ 
w_results = cell(length(lambda_values), 1);
y_pred_results = cell(length(lambda_values), 1);

for k = 1:length(lambda_values)
    lambda = lambda_values(k);
    w = (Phi_train' * Phi_train + lambda * eye(N)) \ (Phi_train' * y_train');
    w_results{k} = w;  % Store weights for different λ
    M = length(x_test);
    Phi_test = zeros(M, N);
    for i = 1:M
        for j = 1:N
            Phi_test(i, j) = exp(-((x_test(i) - x_train(j))^2) / (2 * sigma^2)); % Compute RBF Output on Test Set
        end
    end
    
    y_pred_results{k} = Phi_test * w; % Compute predicted values
end

%% Draw multiple subplots, each corresponding to a different λ
figure;
num_lambda = length(lambda_values);
num_cols = 3;
num_rows = ceil(num_lambda / num_cols);

for k = 1:num_lambda
    subplot(num_rows, num_cols, k);
    hold on;
    plot(x_test, y_test, 'b-', 'LineWidth', 2);
    plot(x_test, y_pred_results{k}, 'r--', 'LineWidth', 2);
    scatter(x_train, y_train, 'ko', 'MarkerFaceColor', 'k');
    legend('True Function', 'RBFN Approximation', 'Training Data', 'FontSize', 4.5);
    xlabel('x');
    ylabel('y');
    title(sprintf('\\lambda = %.1e', lambda_values(k)));
    grid on;
    hold off;
end

sgtitle('Function Approximation using RBFN with Regularization (Exact Interpolation)');
