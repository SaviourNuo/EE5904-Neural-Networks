clc; clear; close all;

%% Load the MNIST dataset
load mnist_m.mat;

%% Set the labels corresponding to 7 and 1 to 1, and the rest to 0 (Matric number: A0313771H)
class1 = 7;
class2 = 1;

% Data processing for training and testing
Train_ClassLabel = zeros(size(train_classlabel));
Train_ClassLabel(train_classlabel == class1 | train_classlabel == class2) = 1;
Test_ClassLabel = zeros(size(test_classlabel));
Test_ClassLabel(test_classlabel == class1 | test_classlabel == class2) = 1;

% Convert the label to double data type to avoid warnings
Train_ClassLabel = double(Train_ClassLabel);
Test_ClassLabel = double(Test_ClassLabel);

Train_Data = train_data;
Test_Data = test_data;

%% Select 100 random centers from the training data
num_centers = 100; % Number of randomly selected centers
center_indices = randperm(size(Train_Data, 2), num_centers); % Randomly select 100 indices
centers = Train_Data(:, center_indices); % Select the corresponding 100 points in Train_Data as the centers of RBF by indices

%% Compute σ based on max distance
d_max = max(pdist(centers')); 
sigma_adaptive = d_max / sqrt(2 * num_centers);  % Adaptive sigma for RBF

%% Define sigma values (including adaptive sigma)
sigma_values = [sigma_adaptive, 0.1, 1, 10, 100, 1000, 10000];

%% Iterate over different sigma values and compute accuracy
N = length(Train_ClassLabel);
M = length(Test_ClassLabel);

for s = 1:length(sigma_values)
    sigma = sigma_values(s);

    % Compute RBF kernel matrix for training set
    Phi_train = zeros(N, num_centers);
    for i = 1:N
        for j = 1:num_centers
            Phi_train(i, j) = exp(-((Train_Data(:,i) - centers(:,j))' * (Train_Data(:,i) - centers(:,j))) / (2 * sigma^2));
        end
    end

    % Compute weights
    w = Phi_train \ Train_ClassLabel';

    % Compute RBF kernel matrix for test set
    Phi_test = zeros(M, num_centers);
    for i = 1:M
        for j = 1:num_centers
            Phi_test(i, j) = exp(-((Test_Data(:,i) - centers(:,j))' * (Test_Data(:,i) - centers(:,j))) / (2 * sigma^2));
        end
    end

    % Compute predicted output
    TrPred = Phi_train * w;
    TePred = Phi_test * w;

%% Compute classification accuracy for different thresholds
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrN = length(Train_ClassLabel);
    TeN = length(Test_ClassLabel);

    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(Train_ClassLabel(TrPred<t)==0) + sum(Train_ClassLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(Test_ClassLabel(TePred<t)==0) + sum(Test_ClassLabel(TePred>=t)==1)) / TeN;
    end

%% Plot threshold vs. accuracy curve
    figure;
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');
    xlabel('Threshold');
    ylabel('Accuracy');
    if sigma == sigma_adaptive
        title(sprintf('Fixed Centers (Adaptive Sigma = %.2f)', sigma));
    else
        title(sprintf('Fixed Centers (Width = %.2f)', sigma));
    end
    grid on;
end

