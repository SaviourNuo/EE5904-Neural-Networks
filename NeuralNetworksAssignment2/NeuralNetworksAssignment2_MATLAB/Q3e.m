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

% Create empty cell arrays to store the image paths of the training set and test set
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

% Get the number of images 
numTrain = numel(train_img.Files);
numTest = numel(test_img.Files);

% Store the pixel value of each image
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

% Get the global pixel mean and standard deviation
global_mean = mean(all_pixels);
global_std = std(all_pixels);

disp(['Global Mean: ', num2str(global_mean)]);
disp(['Global Standard Deviation: ', num2str(global_std)]);

X_train = zeros(1024, numTrain);
X_test = zeros(1024, numTest);

for i = 1:numTrain
    img = readimage(train_img, i); % Read training set images
    img_gray = rgb2gray(img); % Convert to grayscale
    img_gray = double(img_gray);
    % Normalization by subtracting the mean value from each image and divide each image by the standard deviation.
    img_gray = (img_gray - global_mean) / global_std;
    X_train(:, i) = img_gray(:); % Flatten to 1024×1 and store in matrix
end

for i = 1:numTest
    img = readimage(test_img, i); 
    img_gray = rgb2gray(img); 
    img_gray = double(img_gray);
    img_gray = (img_gray - global_mean) / global_std;
    X_test(:, i) = img_gray(:); 
end

% Convert data format (suitable for adapt training)
X_train_c = num2cell(X_train, 1); 
Y_train_c = num2cell(train_labels', 1);
X_test_c = num2cell(X_test, 1); 
Y_test_c = num2cell(test_labels', 1);

% Create an MLP
hidden_neurons = 100;
net = patternnet(hidden_neurons);

% Set hyperparameters
net.trainFcn = 'traingdx';
net.performFcn = 'mse';
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.01;
net.performParam.regularization = 0.25;
net.trainParam.showWindow = true;
num_epochs = 1000;
train_acc = zeros(num_epochs, 1);
test_acc = zeros(num_epochs, 1);

% Disable data set splitting to ensure that all data is used for training
net.divideFcn = 'dividetrain';

for epoch = 1:num_epochs
    disp(['Epoch ', num2str(epoch), ' / ', num2str(num_epochs)]);
    
    % Randomly shuffle training set
    idx = randperm(numTrain);
    X_train_c = X_train_c(:, idx);
    Y_train_c = Y_train_c(:, idx);
    
    % Update weights sample by sample
    net = adapt(net, X_train_c, Y_train_c);

    % Evaluate classification accuracy on the training set
    pred_train = round(net(X_train));
    train_acc(epoch) = sum(pred_train == train_labels') / numel(train_labels) * 100;

    % Evaluate classification accuracy on the test set
    pred_test = round(net(X_test));
    test_acc(epoch) = sum(pred_test == test_labels') / numel(test_labels) * 100;
end

% Plot the curves for training and test classification accuracy
figure;
plot(1:num_epochs, train_acc, 'b-', 'LineWidth', 2);
hold on;
plot(1:num_epochs, test_acc, 'r-', 'LineWidth', 2);
xlabel('Epochs');
ylabel('Accuracy (%)');
title('Sequential Mode Training Accuracy');
legend('Training Accuracy', 'Test Accuracy');
grid on;

% Compare the results
disp('Comparing Sequential Mode and Batch Mode:');
disp('Batch Mode Training Accuracy: XX% (请手动填入)');
disp('Batch Mode Test Accuracy: XX% (请手动填入)');
disp('Sequential Mode Training Accuracy: ' + string(train_acc(end)) + '%');
disp('Sequential Mode Test Accuracy: ' + string(test_acc(end)) + '%');
