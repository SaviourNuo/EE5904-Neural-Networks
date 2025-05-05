clc;
clear;
close all;

% Folder path
deer_folder = './deer';
ship_folder = './ship';

% Number of images per class for training
num_train_img = 450;

% Traverse all the pictures in the folder
deer_img = dir(fullfile(deer_folder, '*.jpg'));
ship_img = dir(fullfile(ship_folder, '*.jpg'));

% Separate training set and test set of the deer pictures
deer_train = {};
ship_train = {};
deer_test = {};
ship_test = {};

% Separate training set and test set of the deer pictures
for i = 1:length(deer_img)
    [~, name, ~] = fileparts(deer_img(i).name);
    num = str2double(name);
    if num < num_train_img
        deer_train{end + 1} = fullfile(deer_folder, deer_img(i).name);
    else
        deer_test{end + 1} = fullfile(deer_folder, deer_img(i).name);
    end
end

% Separate training set and test set of the ship pictures
for i = 1:length(ship_img)
    [~, name, ~] = fileparts(ship_img(i).name);
    num = str2double(name);
    if num < num_train_img
        ship_train{end + 1} = fullfile(ship_folder, ship_img(i).name);
    else
        ship_test{end + 1} = fullfile(ship_folder, ship_img(i).name);
    end
end

% Merge the training and test sets separately and correspond to the correct labels
train_files = [deer_train, ship_train];
test_files = [deer_test, ship_test];
train_labels = [zeros(length(deer_train), 1); ones(length(ship_train), 1)];
test_labels = [zeros(length(deer_test), 1); ones(length(ship_test), 1)];

% Randomly shuffle the training set to improve generalization ability
temp_train = [train_files', num2cell(train_labels)];
temp_train = temp_train(randperm(size(temp_train, 1)), :);
train_files = temp_train(:, 1)';
train_labels = cell2mat(temp_train(:, 2));

% Randomly shuffle the test set
temp_test = [test_files', num2cell(test_labels)];
temp_test = temp_test(randperm(size(temp_test, 1)), :);
test_files = temp_test(:, 1)';
test_labels = cell2mat(temp_test(:, 2));

% Load training and test set images
train_img = imageDatastore(train_files);
test_img = imageDatastore(test_files);

numTrain = numel(train_img.Files);
numTest = numel(test_img.Files);

% Get the global pixel mean and standard deviation
all_pixels = [];

for i = 1:numTrain
    img = readimage(train_img, i);
    img_gray = rgb2gray(img);
    all_pixels = [all_pixels; double(img_gray(:))];
end

for i = 1:numTest
    img = readimage(test_img, i);
    img_gray = rgb2gray(img);
    all_pixels = [all_pixels; double(img_gray(:))];
end

global_mean = mean(all_pixels);
global_std = std(all_pixels);

disp(['Global Mean: ', num2str(global_mean)]);
disp(['Global Standard Deviation: ', num2str(global_std)]);

X_train = zeros(1024, numTrain);
X_test = zeros(1024, numTest);

for i = 1:numTrain
    img = readimage(train_img, i);
    img_gray = rgb2gray(img);
    img_gray = double(img_gray);
    img_gray = (img_gray - global_mean) / global_std;
    X_train(:, i) = img_gray(:);
end

for i = 1:numTest
    img = readimage(test_img, i);
    img_gray = rgb2gray(img);
    img_gray = double(img_gray);
    img_gray = (img_gray - global_mean) / global_std;
    X_test(:, i) = img_gray(:);
end

% Record the test set accuracy of different regularization coefficients
lambda_values = 0:0.05:0.95;  % λ ranges from 0 to 0.95, with every 0.05
test_accuracies = zeros(length(lambda_values), 1);

% Number of hidden neurons
hidden_neurons = 100; 

% Traverse different λ training MLP
for idx = 1:length(lambda_values)
    lambda = lambda_values(idx);  % Take different values ​​of λ
    disp(['Training with λ = ', num2str(lambda)]);

    % Create an MLP
    net = patternnet(hidden_neurons);

    % Set hyperparameters
    net.trainFcn = 'traingdx'; 
    net.performFcn = 'mse';
    net.trainParam.epochs = 1000;
    net.performParam.regularization = lambda; % Set regularization parameters
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.1;
    net.trainParam.showWindow = true;

    % Change label format
    Y_train = full(ind2vec(train_labels' + 1)); % [0,1] → [1 0] & [0 1]
    Y_test = full(ind2vec(test_labels' + 1));

    % Train the MLP
    [net, ~] = train(net, X_train, Y_train);

    % Evaluate classification accuracy on the test set
    Y_pred_test = net(X_test);
    Y_pred_test = vec2ind(Y_pred_test) - 1;
    accuracy_test = sum(Y_pred_test == test_labels') / numel(test_labels) * 100;
    test_accuracies(idx) = accuracy_test;

    disp(['Lambda = ', num2str(lambda), ' -> Test Accuracy: ', num2str(accuracy_test), '%']);
end

% Result visualization
figure;
plot(lambda_values, test_accuracies, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Regularization Coefficient (λ)');
ylabel('Test Accuracy (%)');
title('Effect of Regularization on Test Accuracy');
grid on;
