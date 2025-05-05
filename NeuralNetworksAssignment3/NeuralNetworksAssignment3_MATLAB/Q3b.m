clc; clear; close all;

%% Generate training data
X = randn(800,2); 
s2 = sum(X.^2,2);
trainX = (X .* repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';

%% SOM initialization
grid_size = [8, 8]; % 8×8 grid for output layer
num_neurons = prod(grid_size); % 64 neurons in total, prod is used to calculate the product of each element in grid_size
[grid_x, grid_y] = meshgrid(1:grid_size(1), 1:grid_size(2));
neuron_pos = [grid_x(:), grid_y(:)]; % Store the grid coordinates of each neuron
w = rand(2, num_neurons) * 2 - 1;  % Randomly initialize weights in the range [-1, 1]
N = 500; % Number of training iterations
initial_lr = 0.1; % Initial learning rate
sigma0 = max(grid_size) / 2; % Initial neighborhood width
tau1 = N / log(sigma0); % Time constant τ1
tau2 = N; % Time constant τ2
idx_plot = [10, 20, 50, 100:100:N];
r = 1; % Index of idx_plot

%% Plot the initial state (iterations = 0)
figure;
hold on;
plot(trainX(1,:), trainX(2,:), '.r', 'MarkerSize', 6); % Plot the original data
plot(w(1,:), w(2,:), 'bo', 'MarkerSize', 6, 'LineWidth', 2); % Plot SOM neurons

for i = 1:grid_size(1) % Plot neurons in horizontal direction
    for j = 1:grid_size(2)-1
        idx1 = sub2ind(grid_size, i, j);
        idx2 = sub2ind(grid_size, i, j+1);
        plot([w(1,idx1), w(1,idx2)], [w(2,idx1), w(2,idx2)], 'k-', 'LineWidth', 1.5);
    end
end

for i = 1:grid_size(1)-1 % Plot neurons in vertical direction
    for j = 1:grid_size(2)
        idx1 = sub2ind(grid_size, i, j);
        idx2 = sub2ind(grid_size, i+1, j);
        plot([w(1,idx1), w(1,idx2)], [w(2,idx1), w(2,idx2)], 'k-', 'LineWidth', 1.5);
    end
end

title('SOM Training at Iteration 0');
axis equal;
grid on;
hold off;

%% SOM training and plot mapping results
for t = 1:N
    idx = randi(size(trainX, 2)); 
    sample = trainX(:, idx); % Randomly select a training sample

    distances = sum((w - sample).^2, 1); % Calculate the Euclidean distance between all neurons and the sample
    [~, bmu_idx] = min(distances); % Find the best matching neuron
    
    % Calculate the current learning rate and neighborhood size
    lr = initial_lr * exp(-t / tau2);  
    sigma = sigma0 * exp(-t / tau1); 

    % Calculate the Euclidean distance from all neurons to the BMU
    neuron_distances = sum((neuron_pos - neuron_pos(bmu_idx, :)).^2, 2);
    
    h = exp(-neuron_distances / (2 * sigma^2))'; % Calculate the neighborhood function
    w = w + lr * h .* (sample - w); % Update the weights of all neurons

    if r <= length(idx_plot) && t == idx_plot(r)
        figure;
        hold on;
        plot(trainX(1,:), trainX(2,:), '.r', 'MarkerSize', 6);
        plot(w(1,:), w(2,:), 'bo', 'MarkerSize', 6, 'LineWidth', 2);

        for i = 1:grid_size(1)
            for j = 1:grid_size(2)-1
                idx1 = sub2ind(grid_size, i, j);
                idx2 = sub2ind(grid_size, i, j+1);
                plot([w(1,idx1), w(1,idx2)], [w(2,idx1), w(2,idx2)], 'k-', 'LineWidth', 1.5);
            end
        end

        for i = 1:grid_size(1)-1
            for j = 1:grid_size(2)
                idx1 = sub2ind(grid_size, i, j);
                idx2 = sub2ind(grid_size, i+1, j);
                plot([w(1,idx1), w(1,idx2)], [w(2,idx1), w(2,idx2)], 'k-', 'LineWidth', 1.5);
            end
        end

        title(['SOM Training at Iteration ', num2str(t)]);
        axis equal;
        grid on;
        hold off;
        r = r + 1;
    end
end