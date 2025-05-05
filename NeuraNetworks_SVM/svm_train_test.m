clc; clear; close all;

load("train.mat");
load("test.mat");

%% Standardization on training and test set
mu = mean(train_data, 2);
sigma = std(train_data, 0, 2);
sigma(sigma < 1e-10) = 1;
train_data = (train_data - mu) ./ sigma;
test_data  = (test_data - mu) ./ sigma;

%% Initialize result storage
results = {};

%% === Hard Margin: Linear kernel ===
[g_train, model, K_train] = svm_discriminant(train_data, train_data, train_label, 1e6, 1, 'linear');
y_train_pred = sign(g_train)';
train_acc = mean(y_train_pred == train_label) * 100;

[g_test, ~] = svm_discriminant([], test_data, [], 1e6, 1, 'linear', model);
y_test_pred = sign(g_test)';
test_acc = mean(y_test_pred == test_label) * 100;

admissible = check_admissibility(K_train);
has_hyperplane = ~isempty(model.sv_alpha) && model.exitflag > 0;
results = [results; {'Hard margin with linear kernel', '-', '-', train_acc, test_acc, admissible, has_hyperplane}];

%% === Hard Margin: Polynomial kernel (p = 2~5) ===
for p = 2:5
    [g_train, model, K_train] = svm_discriminant(train_data, train_data, train_label, 1e6, p, 'polynomial');
    y_train_pred = sign(g_train)';
    train_acc = mean(y_train_pred == train_label) * 100;

    [g_test, ~] = svm_discriminant([], test_data, [], 1e6, p, 'polynomial', model);
    y_test_pred = sign(g_test)';
    test_acc = mean(y_test_pred == test_label) * 100;

    admissible = check_admissibility(K_train);
    has_hyperplane = ~isempty(model.sv_alpha) && model.exitflag > 0;
    results = [results; {'Hard margin with polynomial kernel', string(p), '-', train_acc, test_acc, admissible, has_hyperplane}];
end

%% === Soft Margin: Polynomial kernel (p = 1~5, C = 0.1~2.1) ===
C_list = [0.1, 0.6, 1.1, 2.1];
for p = 1:5
    for C = C_list
        [g_train, model, K_train] = svm_discriminant(train_data, train_data, train_label, C, p, 'polynomial');
        y_train_pred = sign(g_train)';
        train_acc = mean(y_train_pred == train_label) * 100;

        [g_test, ~] = svm_discriminant([], test_data, [], C, p, 'polynomial', model);
        y_test_pred = sign(g_test)';
        test_acc = mean(y_test_pred == test_label) * 100;

        admissible = check_admissibility(K_train);
        has_hyperplane = ~isempty(model.sv_alpha) && model.exitflag > 0;
        results = [results; {'Soft margin with polynomial kernel', string(p), string(C), train_acc, test_acc, admissible, has_hyperplane}];
    end
end

%% === Save results to CSV ===
headers = {'Type_of_SVM', 'p', 'C', 'Train_Accuracy', 'Test_Accuracy', 'Admissible', 'Has_Hyperplane'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'svm_results_table.csv');
fprintf("All results saved to 'svm_results_table.csv'\n");

%% === Kernel admissibility checker ===
function admissible = check_admissibility(K)
    if isempty(K)
        admissible = false;
        return;
    end
    K = (K + K') / 2;
    eigvals = eig(K);
    admissible = all(eigvals > -1e-4);  % Allow for small numerical error
end
