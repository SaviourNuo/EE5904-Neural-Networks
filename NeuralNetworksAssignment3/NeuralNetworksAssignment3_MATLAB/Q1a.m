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

%% Solve for weights w
w = Phi_train \ y_train';  % Compute weights using w = inv(Phi) * d

%% Compute RBF Output on Test Set
M = length(x_test);  % Number of test points
Phi_test = zeros(M, N);
for i = 1:M
    for j = 1:N
        Phi_test(i, j) = exp(-((x_test(i) - x_train(j))^2) / (2 * sigma^2));
    end
end
y_pred = Phi_test * w; % Compute predicted values

%% Plot results
figure;
plot(x_test, y_test, 'b-', 'LineWidth', 2); hold on; % True function
plot(x_test, y_pred, 'r--', 'LineWidth', 2);         % RBFN approximation
scatter(x_train, y_train, 'ko', 'MarkerFaceColor', 'k'); % Training points
legend('True Function', 'RBFN Approximation', 'Training Data');
xlabel('x');
ylabel('y');
title('Function Approximation using RBFN (Exact Interpolation)');
grid on;
