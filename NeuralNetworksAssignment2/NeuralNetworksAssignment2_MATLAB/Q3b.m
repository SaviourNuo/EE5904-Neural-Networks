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
for i = 1 : length(deer_img)
    [~, name, ~] = fileparts(deer_img(i).name);
    num = str2double(name);
    if num >= 0 && num < num_train_img
        deer_train{end + 1} = fullfile(deer_folder, deer_img(i).name);
    else
        deer_test{end + 1} = fullfile(deer_folder, deer_img(i).name);
    end
end

% Separate training set and test set of the ship pictures
for i = 1 : length(ship_img)
    [~, name, ~] = fileparts(ship_img(i).name);
    num = str2double(name);
    if num >= 0 && num < num_train_img
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

% Traverse the training set and test set to get all pixel values
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
    img_gray = (img_gray - global_mean) / global_std; % 归一化
    X_test(:, i) = img_gray(:);
end

Y_train = train_labels; 
Y_test = test_labels;

% Create a perceptron
net = perceptron();

% Set hyperparameters
net.trainParam.epochs = 2500;
net.trainParam.lr = 0.01;
net.trainParam.showWindow = true;

% Train the perceptron
[net, tr] = train(net, X_train, Y_train');

% Evaluate classification accuracy on the training set
disp('Perceptron Model Training Completed.');
Y_pred_train = net(X_train);
Y_pred_train = round(Y_pred_train);
accuracy_train = sum(Y_pred_train == Y_train') / numel(Y_train) * 100;
disp(['Train Accuracy: ', num2str(accuracy_train), '%']);

% Evaluate classification accuracy on the test set
Y_pred_test = net(X_test);
Y_pred_test = round(Y_pred_test); 
accuracy_test = sum(Y_pred_test == Y_test') / numel(Y_test) * 100;
disp(['Test Accuracy: ', num2str(accuracy_test), '%']);

% plot the training curve
figure;
plot(1:length(tr.epoch), tr.perf, 'b-', 'LineWidth', 2);
xlabel('Epochs');
ylabel('Training Performance');
title('Training Performance Over Epochs');
grid on;

% Randomly select 9 images from the test set to preview the classification results
figure;
sgtitle('Sample Predictions');
for i = 1:min(9, numTest)
    subplot(3, 3, i);
    img = readimage(test_img, i);
    imshow(img);
    actual = test_labels(i);
    predicted = Y_pred_test(i);
    if actual == 0
        actual_class = 'deer';
    else
        actual_class = 'ship';
    end
    if predicted == 0
        predicted_class = 'deer';
    else
        predicted_class = 'ship';
    end
    title_color = 'green';
    if actual ~= predicted
        title_color = 'red';
    end
    title({['Actual: ' actual_class], ['Pred: ' predicted_class]}, 'Color', title_color);
end
