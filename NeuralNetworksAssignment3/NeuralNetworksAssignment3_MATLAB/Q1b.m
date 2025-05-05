clc; clear; close all;

%% Generate training and test data
x_train = -1.6:0.08:1.6; % Training set with a uniform step length 0.08
y_train = 1.2*sin(pi*x_train) - cos(2.4*pi*x_train) + 0.3*randn(size(x_train)); % Add random noise
x_test = -1.6:0.01:1.6; % Test set with a uniform step length 0.01
y_test = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test); % Ground truth without noise

%% Select random centers
num_centers = 20; % Number of randomly selected centers
center_indices = randperm(length(x_train), num_centers); % Randomly select 20 indices
centers = x_train(center_indices); % Select the corresponding 20 points in x_train as the centers of RBF by indices

%% Compute σ based on max distance
d_max = max(pdist(centers')); % Calculate the maximum distance between the chosen centres
sigma = d_max / sqrt(2 * num_centers); % Adaptive sigma

%% Construct RBF Kernel Matrix
N = length(x_train); % Number of training points
Phi_train = zeros(N, num_centers); % Initialize Gaussian Kernel

for i = 1:N
    for j = 1:num_centers
        Phi_train(i, j) = exp(-((x_train(i) - centers(j))^2) * (num_centers / d_max^2)); % Gaussian Kernel
    end
end

%% Solve for weights w using Least Squares
w = pinv(Phi_train' * Phi_train) * (Phi_train' * y_train');

%% Compute RBF Output on Test Set
M = length(x_test);
Phi_test = zeros(M, num_centers);
for i = 1:M
    for j = 1:num_centers
        Phi_test(i, j) = exp(-((x_test(i) - centers(j))^2) * (num_centers / d_max^2));
    end
end
y_pred = Phi_test * w; % Compute predicted values

%% Plot Results
figure;
plot(x_test, y_test, 'b-', 'LineWidth', 2); hold on;
plot(x_test, y_pred, 'r--', 'LineWidth', 2);
scatter(x_train, y_train, 'ko', 'MarkerFaceColor', 'k');
scatter(centers, zeros(size(centers)), 'ro', 'MarkerFaceColor', 'r', 'SizeData', 100);
legend('True Function', 'RBFN Approximation', 'Training Data', 'Selected Centers');
xlabel('x');
ylabel('y');
title('Function Approximation using RBFN (Fixed Centers Selected at Random)');
grid on;
