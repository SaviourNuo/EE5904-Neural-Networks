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

%% Construct RBF Kernel Matrix for Training Data
N = length(Train_ClassLabel);
sigma = 100; % Standard deviation of 100
Phi_train = zeros(N, N); % Initialize Gaussian Kernel

for i = 1:N
    for j = 1:N
        Phi_train(i, j) = exp(-((Train_Data(:,i) - Train_Data(:,j))' * (Train_Data(:,i) - Train_Data(:,j))) / (2 * sigma^2));
    end
end

%% Calculate the weights under different regularization parameters λ and predict the output of test set
lambda_values = [0, 1e-6, 1e-4, 1e-2, 1, 10, 40]; % Different λ 
w_results = cell(length(lambda_values), 1);
y_pred_results = cell(length(lambda_values), 1);

for k = 1:length(lambda_values)
    lambda = lambda_values(k);

    if lambda == 0

        % Calculate the weight without regularization separately because the calculation formula is different
        w = Phi_train \ Train_ClassLabel';
    else

        % Calculate weights with regularization using LSE
        w = pinv(Phi_train' * Phi_train + lambda * eye(N)) * (Phi_train' * Train_ClassLabel');
    end

    w_results{k} = w;

    % Compute the test set RBF kernel matrix
    M = length(Test_ClassLabel);
    Phi_test = zeros(M, N);

    for i = 1:M
        for j = 1:N
            Phi_test(i, j) = exp(-((Test_Data(:,i) - Train_Data(:,j))' * (Test_Data(:,i) - Train_Data(:,j))) / (2 * sigma^2));
        end
    end

    % Calculate predicted output
    TrPred = Phi_train * w;
    TePred = Phi_test * w;

%% Calculate classification accuracy and visualize
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

    % Plot the accuracy curve
    figure;
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');
    xlabel('Threshold');
    ylabel('Accuracy');
    title(sprintf('RBFN Accuracy vs. Threshold (\\lambda = %.1e)', lambda_values(k)));
    grid on;
    hold off;
end