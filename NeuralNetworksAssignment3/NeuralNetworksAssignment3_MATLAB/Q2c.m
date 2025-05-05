clc; clear; close all;

%% Load the MNIST dataset
load mnist_m.mat;

%% Set the labels corresponding to 7 and 1 to 1, and the rest to 0 (Matric number: A0313771H)
class1 = 7;
class2 = 1;

% Data processing for training and testing
Train_ClassLabel = double(train_classlabel == class1 | train_classlabel == class2);
Test_ClassLabel = double(test_classlabel == class1 | test_classlabel == class2);
Train_Data = train_data;
Test_Data = test_data;

%% Select only class 7 and 1 for K-Means clustering
trainIdx = find(train_classlabel == class1 | train_classlabel == class2);
KMeans_Data = Train_Data(:, trainIdx);
KMeans_Labels = Train_ClassLabel(trainIdx);

%% Apply Manual K-Means Clustering (2 centers)
num_clusters = 2;
max_iters = 20;  % Maximum iterations
rng(1); % Set random seed for reproducibility

% Initialize centers randomly from the selected data
rand_indices = randperm(size(KMeans_Data, 2), num_clusters);
centers = KMeans_Data(:, rand_indices);

% K-Means Iteration
for iter = 1:max_iters
    % Assign each sample to the closest center
    cluster1 = [];
    cluster2 = [];
    
    for i = 1:size(KMeans_Data, 2)
        d1 = norm(KMeans_Data(:, i) - centers(:, 1));
        d2 = norm(KMeans_Data(:, i) - centers(:, 2));
        
        if d1 < d2
            cluster1 = [cluster1, KMeans_Data(:, i)];
        else
            cluster2 = [cluster2, KMeans_Data(:, i)];
        end
    end
    
    % Update centers
    new_center1 = mean(cluster1, 2);
    new_center2 = mean(cluster2, 2);
    
    % Convergence check
    if norm(new_center1 - centers(:, 1)) < 1e-6 && norm(new_center2 - centers(:, 2)) < 1e-6
        break;
    end
    
    centers(:, 1) = new_center1;
    centers(:, 2) = new_center2;
end

%% Compute σ based on max distance
d_max = max(pdist(centers'));  
sigma_adaptive = d_max / sqrt(2 * num_clusters);

%% Construct RBF Kernel Matrix for Training Data
N = length(Train_ClassLabel);
M = length(Test_ClassLabel);
Phi_train = zeros(N, num_clusters);

for i = 1:N
    for j = 1:num_clusters
        Phi_train(i, j) = exp(-((Train_Data(:,i) - centers(:,j))' * (Train_Data(:,i) - centers(:,j))) / (2 * sigma_adaptive^2));
    end
end

%% Compute weights (RBFN Training)
w = Phi_train \ Train_ClassLabel';

%% Construct RBF Kernel Matrix for Test Data
Phi_test = zeros(M, num_clusters);
for i = 1:M
    for j = 1:num_clusters
        Phi_test(i, j) = exp(-((Test_Data(:,i) - centers(:,j))' * (Test_Data(:,i) - centers(:,j))) / (2 * sigma_adaptive^2));
    end
end

%% Compute Predicted output
TrPred = Phi_train * w;
TePred = Phi_test * w;

%% Compute classification accuracy for different thresholds
TrAcc = zeros(1,1000);
TeAcc = zeros(1,1000);
thr = linspace(min(TePred), max(TePred), 1000);
TrN = length(Train_ClassLabel);
TeN = length(Test_ClassLabel);

for i = 1:1000
    t = thr(i);
    TrAcc(i) = (sum(Train_ClassLabel(TrPred<t)==0) + sum(Train_ClassLabel(TrPred>=t)==1)) / TrN;
    TeAcc(i) = (sum(Test_ClassLabel(TePred<t)==0) + sum(Test_ClassLabel(TePred>=t)==1)) / TeN;
end

%% Plot Threshold vs. Accuracy
figure;
plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('Train Accuracy','Test Accuracy');
xlabel('Threshold');
ylabel('Accuracy');
title('K-Means Clustering with RBFN (2 Centers)');
grid on;

%% Compute Mean Images for Each Class (0-9)
unique_classes = unique(train_classlabel);
num_classes = length(unique_classes);
mean_images = zeros(size(train_data,1), num_classes); % Store mean of each class

for i = 1:num_classes
    class = unique_classes(i);
    mean_images(:, i) = mean(train_data(:, train_classlabel == class), 2);
end

%% Visualize the Mean Images of Each Class (0-9)
figure;
for i = 1:num_classes
    subplot(2,5,i);
    imshow(reshape(mean_images(:,i), 28, 28), []);
    title(['Mean of Class ', num2str(unique_classes(i))]);
end
sgtitle('Mean Images of Each Class (0-9)');

%% Visualize the Centers and Class Means for 7 & 1
% Compute mean images for class 7 and 1
mean_class1 = mean_images(:, unique_classes == class1);
mean_class2 = mean_images(:, unique_classes == class2);

% Compute Euclidean distance between cluster centers and mean images
dist1 = norm(centers(:,1) - mean_class1); 
dist2 = norm(centers(:,1) - mean_class2);

% Assign correct labels to cluster centers
if dist1 < dist2
    center1_label = class1;
    center2_label = class2;
else
    center1_label = class2;
    center2_label = class1;
end

%% Visualize K-Means Centers Compared to Class Means
figure;
subplot(2,2,1);
imshow(reshape(mean_class1, 28, 28), []);
title(['Mean of Class ', num2str(class1)]);

subplot(2,2,2);
imshow(reshape(mean_class2, 28, 28), []);
title(['Mean of Class ', num2str(class2)]);

subplot(2,2,3);
imshow(reshape(centers(:,2), 28, 28), []);
title(['K-Means Center for ', num2str(center2_label)]);

subplot(2,2,4);
imshow(reshape(centers(:,1), 28, 28), []);
title(['K-Means Center for ', num2str(center1_label)]);

sgtitle('Comparison of K-Means Centers with Class Means');

