clear; clc; close all;

%% Load the Digits dataset
load('Digits.mat')

%% Select training and test data (Metric number: A0313771H) (Omit class 1 and 2)
Train_idx = find(train_classlabel == 0 | train_classlabel == 3 | train_classlabel == 4);
Train_data = train_data(:, Train_idx);
Train_label = train_classlabel(Train_idx);

Test_idx = find(test_classlabel == 0 | test_classlabel == 3 | test_classlabel == 4);
Test_data = test_data(:, Test_idx);
Test_label = test_classlabel(Test_idx);

Train_Number = size(Train_data, 2);
Test_Number = size(Test_data, 2);

%% SOM initialization
N = 1000; % Number of training iterations
t = 0; % Current iterations
idx_record = [0, 10, 20, 50, 100:100:N]; 
r = 1; % Index of idx_plot
num_neurons = 100; % 2D SOM 100 neurons
w = rand(784, num_neurons);
sigma0 = num_neurons / 2; % Initial neighborhood width
initial_lr = 0.1; % Initial learning rate
tau = N / log(sigma0); % Time constant
Train_accuracy = zeros(1, size(idx_record, 2));
Test_accuracy = zeros(1, size(idx_record, 2)); 

%% SOM training
while t <= N
    idx = randi(Train_Number);
    sample = Train_data(:, idx); % Randomly select a training sample

    distances = sum((w - sample).^2, 1); % Calculate the Euclidean distance between all neurons and the sample
    [~, bmu_idx] = min(distances); % Find the best matching neuron

    % Calculate the current learning rate and neighborhood size
    sigma = sigma0 * exp(-t / tau);
    lr = initial_lr * exp(-t / N);

    for i = 1 : num_neurons
        % Calculate the topological distance from the current neuron to the BMU
        d = (fix((i - 1) / 10) - fix((bmu_idx - 1) / 10)) ^ 2 + (mod(i - 1, 10) - mod(bmu_idx - 1, 10)) ^ 2;
        h = exp(-d^2 / (2 * sigma^2)); % Calculate the neighborhood function
        w(:, i) = w(:, i) + lr * h * (sample - w(:, i)); % Update the weights
    end

    if t == idx_record(r)
        % Count the number of times each neuron is activated by samples of different classes
        vote = zeros(5, num_neurons);
        for i = 1 : Train_Number
            sample = Train_data(:, i); % Get a training sample
            [~, bmu_idx] = min(sum((sample - w) .^ 2)); % Find the best matching neuron
            vote(Train_label(i) + 1, bmu_idx) = vote(Train_label(i) + 1, bmu_idx) + 1; % Vote counts + 1
        end

        neurons_label = zeros(1, num_neurons);
        neurons_val = zeros(1, num_neurons);

        for i = 1 : num_neurons
            [val, bmu_idx] = max(vote(:, i)); % Find the classes with the most votes
            neurons_label(i) = bmu_idx - 1; % Record the label of the neurons
            neurons_val(i) = val; % Record the votes
        end

        % Calculate the accuracy of the test set
        for i = 1 : Test_Number
            sample = Test_data(:, i);
            [~, bmu_idx] = min(sum((sample - w) .^ 2));
            Test_accuracy(r) = Test_accuracy(r) + (neurons_label(bmu_idx) == Test_label(i));
        end

        % Calculate the accuracy of the training set
        for i = 1 : Train_Number
            sample = Train_data(:, i);
            [~, bmu_idx] = min(sum((sample - w) .^ 2));
            Train_accuracy(r) = Train_accuracy(r) + (neurons_label(bmu_idx) == Train_label(i));
        end

        Test_accuracy(r) = Test_accuracy(r) / Test_Number;
        Train_accuracy(r) = Train_accuracy(r) / Train_Number;
        r = r + 1;
    end

    t = t + 1;
end

%% Plot the results

% Weights
Trained_weights = [];

for i = 0 : 9
    Weights_row = [];

    for j = 1 : 10
        Weights_row = [Weights_row, reshape(w(:, i*10+j), 28, 28)];
    end

    Trained_weights = [Trained_weights; Weights_row];
end

figure;
imshow(imresize(Trained_weights, 3));
title('Weights Visualization');


% Conceptual map
neurons_label = reshape(neurons_label, 10, 10)';
neurons_val = neurons_val/max(neurons_val); % Store the confidence of each neuron (normalized vote counts)
neurons_val = reshape(neurons_val, [10,10])';

figure;
img = imagesc(neurons_label);
img.AlphaData = neurons_val;
% Set the transparency. The smaller the value of neurons_val, the more transparent it is, meaning that the neuron has fewer votes.

for i = [0, 3, 4]
    neurons_label(neurons_label == i) = num2str(i);
end

label = num2str(neurons_label, '%s');       
[x, y] = meshgrid(1:10);  
hStrings = text(x(:), y(:), label(:), 'HorizontalAlignment', 'center'); % Label each neuron's class
title('Conceptual Map');

% Accuracy curves of training set and test set
figure;
hold on;
plot(idx_record, Train_accuracy, 'linewidth', 2);
plot(idx_record, Test_accuracy, 'linewidth', 2);
hold off;
legend('Train Accuracy','Test Accuracy', 'Location', 'southeast');
xlabel('Iterations')
ylabel('Accuracy')
title('Accuracy Curves of Training Set and Test Set');
