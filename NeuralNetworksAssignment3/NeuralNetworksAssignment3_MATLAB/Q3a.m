clc; clear; close all;

%% Generate training data (sinc function)
x = linspace(-pi, pi, 400); 
trainX = [x; sinc(x)];
num_neurons = 40; % 1D SOM 40 neurons

%% Initialize the weights of 40 neurons (randomly distributed)
w = rand(2, num_neurons) * 2 * pi - pi; % Randomly initialize weights in the range [-π, π]
N = 500; % Number of training iterations
idx_plot = [10, 20, 50, 100:100:N]; % Plot when the appropriate number of iterations is reached
r = 1; % Index of idx_plot
initial_lr = 0.1; % Initial learning rate
sigma0 = num_neurons / 2; % Initial neighborhood width
tau = N / log(sigma0); % Time constant

%% Plot the initial state (iterations = 0)
figure;
hold on;
plot(trainX(1,:), trainX(2,:), '.r', 'MarkerSize', 8); % Plot the original function
plot(w(1,:), w(2,:), 'bo-', 'LineWidth', 2, 'MarkerSize', 6); % Plot SOM neurons

for j = 1:num_neurons-1
    plot([w(1,j), w(1,j+1)], [w(2,j), w(2,j+1)], 'k-', 'LineWidth', 1.5); % Connect adjacent SOM neurons
end

xlabel('x');
ylabel('sinc(x)');
title('SOM Mapping of 40 Neurons to sinc Function, Iteration 0');
grid on;
hold off;

%% SOM training and plot mapping results
for t = 1:N
    idx = randi(size(trainX, 2));
    sample = trainX(:, idx); % Randomly select a training sample

    distances = sum((w - sample).^2, 1); % Calculate the Euclidean distance between all neurons and the sample
    [~, bmu_idx] = min(distances); % Find the best matching neuron
    
    % Calculate the current learning rate and neighborhood size
    lr = initial_lr * exp(-t / N);
    sigma = sigma0 * exp(-t / tau);
    
    % Update the weights of all neurons
    for j = 1:num_neurons
        % Calculate the topological distance from the current neuron to the BMU
        d = abs(j - bmu_idx);
        
        % Calculate the neighborhood function
        h = exp(-d^2 / (2 * sigma^2));
        
        w(:, j) = w(:, j) + lr * h * (sample - w(:, j));
    end
    
    if r <= length(idx_plot) && t == idx_plot(r)      
        figure;
        hold on;
        plot(trainX(1,:), trainX(2,:), '.r', 'MarkerSize', 8); 
        plot(w(1,:), w(2,:), 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
        for j = 1:num_neurons-1
            plot([w(1,j), w(1,j+1)], [w(2,j), w(2,j+1)], 'k-', 'LineWidth', 1.5);
        end
        xlabel('x');
        ylabel('sinc(x)');
        title(['SOM Mapping of 40 Neurons to sinc Function, Iteration ', num2str(t)]); 
        grid on;
        hold off;
        r = r + 1;
    end
end